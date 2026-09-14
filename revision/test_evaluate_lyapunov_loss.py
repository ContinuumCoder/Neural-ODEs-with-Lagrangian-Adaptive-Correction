#!/usr/bin/env python3
"""Reproduce evaluator diagnostics on synthetic states without importing the study.

Example:
    python revision/test_evaluate_lyapunov_loss.py --audit-json /path/to/audit.json
Only the evaluator's diagnostic functions are extracted by AST. No selection
lock, dataset, or checkpoint is required. All calculations use CPU float64.
"""
import argparse
import unittest
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / 'revision/evaluate_lyapunov_loss.py'
RTOL, ATOL = 1e-12, 1e-10
CHUNKS = (511, 1024, 4096, 20000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_function(tree, name, chunk_size=1024):
    node = copy.deepcopy(next(n for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    replacements = []

    class ChunkSize(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            if (isinstance(call.func, ast.Attribute) and call.func.attr == 'split'
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == 1024):
                call.args[0] = ast.copy_location(ast.Constant(chunk_size), call.args[0])
                replacements.append(True)
            return call

    node = ChunkSize().visit(node)
    if name == 'full_diagnostics' and len(replacements) != 1:
        raise AssertionError('Expected exactly one 1024-state diagnostic chunk loop')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(EVALUATOR), 'exec'), namespace)
    return namespace[name]


class ToyConstraint:
    """Two squared-hinge envelope constraints with an analytic gradient."""
    def k(self, z):
        return torch.stack([torch.relu(z[..., 2] - .4),
                            torch.relu(z[..., :2].square().sum(-1) - 2)], -1)


def analytic_oracle(z, field_net, gain_net, rho, constraint):
    with torch.no_grad():
        k = constraint.k(z)
        v = .5 * k.square().sum(-1)
        u = torch.cat([2 * k[:, 1, None] * z[:, :2], k[:, :1]], -1)
        clipped_direction = (2 * u).clamp(-5, 5)
        gate = torch.tanh(k.square().sum(-1).sqrt())[:, None]
        field = field_net(z) - rho * gain_net(z) * gate * clipped_direction
        dv = (u * field).sum(-1)
        signed = dv + .1 * v
        raw = signed.clamp_min(0)
        normalized = (raw / (1 + v)).square()
        active = v > 1e-12
        result = {
            'sample_count': len(v), 'active_count': int(active.sum()),
            'active_fraction': float(active.double().mean()),
            'V_mean': float(v.mean()), 'dotV_mean': float(dv.mean()),
            'dotV_max': float(dv.max()), 'dotV_min': float(dv.min()),
            'raw_R_mean': float(raw.mean()), 'raw_R_max': float(raw.max()),
            'normalized_residual_sq_mean': float(normalized.mean()),
            'normalized_residual_sq_max': float(normalized.max()),
            'residual_satisfied_fraction': float((signed <= 1e-10).double().mean()),
            'active_normalized_residual_sq_mean': float(normalized[active].mean()),
            'active_raw_R_mean': float(raw[active].mean()),
            'active_raw_R_max': float(raw[active].max()),
            'active_residual_satisfied_fraction': float((signed[active] <= 1e-10).double().mean()),
        }
        clipped_states = int(((2 * u).abs() > 5).any(-1).sum())
    return result, clipped_states


def check_empty_paired_branch(tree):
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'S_paired_differences'):
                    assignments.append(node.value)
    if len(assignments) != 1:
        raise AssertionError('Expected one paired-score difference expression')
    expression = ast.fix_missing_locations(ast.Expression(copy.deepcopy(assignments[0])))
    namespace = {'br': [{'S': None}, {'S': 2.}, {'S': 2.}],
                 'sr': [{'S': None}, {'S': None}, {'S': 3.}]}
    actual = eval(compile(expression, str(EVALUATOR), 'eval'), namespace)
    if actual != [None, None, 1.]:
        raise AssertionError(f'Empty-anchor paired-score regression: {actual}')
    return {'case': 'paired_score_empty_branch', 'actual': actual, 'passed': True}


def run_checks():
    torch.set_num_threads(2)
    evaluator_hash = sha(EVALUATOR)
    tree = ast.parse(EVALUATOR.read_text())
    sys.path.insert(0, str(REPO))
    checks = []
    with patch('torch.load', side_effect=AssertionError('Toy audit forbids dataset/checkpoint loading')):
        from nodesac.core.nodesac import ClosedLoopDynamics, GainNet
        constraint = ToyConstraint()
        field_net = nn.Linear(3, 3, dtype=torch.float64)
        with torch.no_grad():
            field_net.weight.copy_(torch.eye(3, dtype=torch.float64))
            field_net.bias.fill_(.25)
        torch.manual_seed(19)
        gain_net = GainNet(3, (5,)).double().eval()
        dyn = ClosedLoopDynamics(field_net, gain_net, constraint, correction_scale=.3).eval()
        t = torch.linspace(-4, 4, 12800, dtype=torch.float64)
        z = torch.stack([t, 2 * torch.sin(t), .4 + 4 * torch.cos(3 * t)], -1)
        expected, clipped_states = analytic_oracle(z, field_net, gain_net, .3, constraint)
        if clipped_states == 0 or not 0 < expected['active_count'] < 12800:
            raise AssertionError('Toy must cover clipping and active/inactive states')
        for chunk in CHUNKS:
            actual = extract_function(tree, 'full_diagnostics', chunk)(
                dyn, z.reshape(128, 100, 3), constraint)
            if actual.keys() != expected.keys():
                raise AssertionError('Diagnostic fields differ from the analytic oracle')
            for key in expected:
                torch.testing.assert_close(
                    torch.tensor(actual[key], dtype=torch.float64),
                    torch.tensor(expected[key], dtype=torch.float64),
                    rtol=RTOL, atol=ATOL, msg=key)
            checks.append({'case': 'analytic_chunk_comparison', 'chunk_size': chunk,
                           'sample_count': actual['sample_count'],
                           'active_count': actual['active_count'],
                           'clipped_states': clipped_states,
                           'checked_fields': len(expected), 'passed': True})
        diagnostic = extract_function(tree, 'full_diagnostics')
        score = extract_function(tree, 'score')
        for label, point in [('interior', [0., 0., 0.]), ('boundary', [1., 1., .4])]:
            states = torch.tensor(point, dtype=torch.float64).repeat(12800, 1).reshape(128, 100, 3)
            result = diagnostic(dyn, states, constraint)
            if not (result['sample_count'] == 12800 and result['active_count'] == 0
                    and result['raw_R_max'] == 0 and result['dotV_max'] == 0):
                raise AssertionError(f'Unexpected {label} diagnostic: {result}')
            nullable = ['active_normalized_residual_sq_mean', 'active_raw_R_mean',
                        'active_raw_R_max', 'active_residual_satisfied_fraction']
            if any(result[key] is not None for key in nullable):
                raise AssertionError('Empty-active statistics must be None')
            if score({'reference': result, 'rollout': result}) is not None:
                raise AssertionError('Empty-active selection score must be None')
            checks.append({'case': label, 'sample_count': 12800, 'active_count': 0,
                           'score': None, 'zero_residual': True, 'passed': True})
        checks.append(check_empty_paired_branch(tree))
    if sha(EVALUATOR) != evaluator_hash:
        raise AssertionError('Evaluator source changed during audit')
    return {'status': 'passed', 'audit': 'synthetic evaluator diagnostics',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'test_source_sha256': sha(__file__), 'evaluator_source_sha256': evaluator_hash,
            'closed_loop_source_sha256': sha(REPO / 'nodesac/core/nodesac.py'),
            'python': sys.version.split()[0], 'torch': torch.__version__,
            'device': 'cpu', 'dtype': 'float64', 'rtol': RTOL, 'atol': ATOL,
            'dataset_or_checkpoint_loads': 0, 'selection_lock_required': False,
            'checks': checks}


class EvaluatorChecks(unittest.TestCase):
    def test_diagnostics_against_analytic_oracle(self):
        audit = run_checks()
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["dataset_or_checkpoint_loads"], 0)
        self.assertTrue(all(check["passed"] for check in audit["checks"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-json', type=Path)
    args = parser.parse_args()
    audit = run_checks()
    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': audit['status'], 'checks': len(audit['checks']),
                      'audit_json': str(args.audit_json) if args.audit_json is not None else None},
                     allow_nan=False), flush=True)
