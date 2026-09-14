#!/usr/bin/env python3
"""Meaningful preflight checks; no test split or experiment evaluation entrypoint."""
import copy,sys,time,unittest,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tune_lyapunov_loss as study

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.info = None
        if os.environ.get("NODE_LAC_ARCHIVE_TESTS") == "1":
            try:
                cls.info = study.prepared()
            except (FileNotFoundError, PermissionError):
                pass
    def setUp(self):
        if self._testMethodName in {
                "test_train_only_loader", "test_zero_weight_copied_loop_exact",
                "test_selected_parent_replay"}:
            if self.info is None:
                self.skipTest("Optional historical archive check; set NODE_LAC_ARCHIVE_TESTS=1 with its inputs")
            if torch.cuda.device_count() < 2:
                self.skipTest("Historical checkpoint replay requires CUDA device 1")
    def test_train_only_loader(self):
        original=torch.load;allowed={study.prior.CACHE/f"{s}.pt" for s in study.frozen.SYSTEMS}
        def guarded(path,*a,**kw):
            if Path(path) not in allowed:raise AssertionError("Non train-only file opened")
            return original(path,*a,**kw)
        with patch.object(torch,"load",side_effect=guarded),patch.object(study.frozen,"load_data",side_effect=AssertionError("Test-capable loader forbidden")):
            for system in study.frozen.SYSTEMS:
                tr,val,times,constraint,prov=study.load_train_validation(system,"cuda:1",self.info)
                self.assertEqual(tuple(tr.shape[:2]),(102,100));self.assertEqual(tuple(val.shape[:2]),(26,100))
                study.prior.verify_preprocessing(prov,self.info["reuse"][f"{system}/42"]["preprocessing"])
    def test_field_gradient_and_sampling(self):
        class Constraint:
            def k(self,x):return torch.relu(x-1)
        constraint=Constraint()
        layer=nn.Linear(2,2,bias=False,dtype=torch.float64)
        with torch.no_grad():layer.weight.copy_(10*torch.eye(2,dtype=torch.float64))
        node=SimpleNamespace(f=layer)
        gain=nn.Sequential(nn.Linear(2,1,dtype=torch.float64),nn.Softplus())
        with torch.no_grad():gain[0].weight.zero_();gain[0].bias.zero_()
        x=torch.tensor([[0.,0.],[1.,1.],[2.,3.],[10.,20.]],dtype=torch.float64,requires_grad=True)
        expected_u=torch.relu(x.detach()-1)
        for rho in [0.,.3,.75,2.]:
            actual=study.residual_values(node,gain,x,constraint,rho,True)
            old=study.ClosedLoopDynamics(node.f,gain,constraint,correction_scale=rho)(torch.tensor(0.),x.detach())
            torch.testing.assert_close(actual["field"],old,atol=0,rtol=0)
            torch.testing.assert_close(actual["u"],expected_u,atol=0,rtol=0)
            torch.testing.assert_close(actual["dotV"],(expected_u*old).sum(-1),atol=0,rtol=0)
        loss=study.loss_residual(node,gain,x,x*2,constraint,.3)
        layer.zero_grad();gain.zero_grad();loss.backward()
        self.assertIsNone(x.grad)
        for model in [layer,gain]:
            grads=[p.grad for p in model.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        large=torch.arange(6400*2,dtype=torch.float64).reshape(64,100,2).requires_grad_(True)
        rng=torch.get_rng_state().clone()
        sample=study.sample_states(large)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        expected=torch.linspace(0,6399,512,dtype=torch.float64).long()
        self.assertTrue(torch.equal(sample,large.reshape(-1,2)[expected]))
        self.assertEqual(tuple(sample.shape),(512,2));self.assertFalse(sample.requires_grad)
        small=large[:1,:10];self.assertTrue(torch.equal(study.sample_states(small),small.reshape(-1,2)))
    def test_zero_weight_copied_loop_exact(self):
        system=study.frozen.SYSTEMS[0]
        tr,val,times,constraint,_=study.load_train_validation(system,"cuda:1",self.info)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        a=root/"frozen";b=root/"copy"
        a.mkdir(parents=True,exist_ok=True);b.mkdir(parents=True,exist_ok=True)
        study.frozen.seed_everything(987)
        study.frozen.train_lac(tr,val,times,constraint,"cuda:1",2,a,study.options(system))
        study.frozen.seed_everything(987)
        with patch.object(study,"loss_residual",side_effect=AssertionError("Zero lambda evaluated residual")):
            study.train_lac(tr,val,times,constraint,"cuda:1",2,b,study.options(system),0.)
        old=torch.load(a/"model.pt",map_location="cpu",weights_only=True)
        new=torch.load(b/"model.pt",map_location="cpu",weights_only=True)
        for name in ["node","gain"]:
            self.assertEqual(old[name].keys(),new[name].keys())
            for key in old[name]:self.assertTrue(torch.equal(old[name][key],new[name][key]),(name,key))
        self.assertTrue(torch.equal(old["log_mu"],new["log_mu"]))
        self.assertEqual(old["selected_scale"],new["selected_scale"])
        self.assertEqual(study.read(a/"history.json"),study.read(b/"history.json"))
    def test_selected_parent_replay(self):
        evidence=[]
        for system in study.frozen.SYSTEMS:
            tr,val,times,constraint,_=study.load_train_validation(system,"cuda:1",self.info)
            for seed in study.SEEDS:
                base=self.info["reuse"][f"{system}/{seed}"]
                node,gain,_=study.model_from_checkpoint(base["checkpoint"]["path"],tr.shape[-1],constraint,"cuda:1",system)
                rows=study.score_rhos(node,gain,val,times,constraint)
                for a,b in zip(rows,base["rho_scores"]):
                    self.assertEqual(a["rho"],b["rho"]);self.assertEqual(a["validation"],b["validation"])
                    for scope in ["reference","rollout"]:self.assertEqual(a["diagnostics"][scope]["sample_count"],2600)
                evidence.append({"system":system,"seed":seed,"rho_count":13,"all_prediction_metric_differences":0})
        self.assertEqual(len(evidence), 12)
    def test_selector_guards(self):
        rows=[{"rho":rho,"finite_mse":False,"validation":{"MSE":None}} for rho in study.RHOS]
        with self.assertRaises(FloatingPointError):study.choose(rows)
        with self.assertRaises(ValueError):study.choose(rows[:-1])
        with self.assertRaises(ValueError):study.choose(rows[:-1]+[rows[0]])
        system=study.frozen.SYSTEMS[0]
        records=[]
        for i in range(4):
            for seed in study.SEEDS:
                records.append({"candidate_index":i,"seed":seed,"status":"complete",
                  "checkpoint":{"path":"unused","sha256":"unused"},
                  "selected":{"rho":.3,"validation":{"MSE":1.,"CE":None},"S":1./(i+1),
                              "diagnostics":{"reference":{"active_count":3}}}})
        self.assertEqual(study.select_system(records,system)["selected"]["candidate_index"],3)
        bad=copy.deepcopy(records);bad[-1]["selected"]["validation"]["MSE"]=1.11
        self.assertEqual(study.select_system(bad,system)["selected"]["candidate_index"],2)
        bad=copy.deepcopy(records)
        for r in bad[9:]:r["selected"]["validation"]["MSE"]=1.06
        self.assertEqual(study.select_system(bad,system)["selected"]["candidate_index"],2)
        bad=copy.deepcopy(records);bad[-1]["status"]="failed"
        self.assertEqual(study.select_system(bad,system)["selected"]["candidate_index"],2)
        with self.assertRaises(ValueError):study.select_system(records[:-1],system)
        empty=copy.deepcopy(records)
        for r in empty:r["selected"]["diagnostics"]["reference"]["active_count"]=0;r["selected"]["S"]=None
        self.assertEqual(study.select_system(empty,system)["selected"]["candidate_index"],0)

if __name__ == "__main__":
    unittest.main()
