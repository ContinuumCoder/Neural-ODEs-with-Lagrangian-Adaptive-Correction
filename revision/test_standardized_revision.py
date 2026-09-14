"""Targeted checks for the final standardized-coordinate revision."""
import sys
import os
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import torch
import run_standardized_revision as runner
from run_standardized_revision import StandardizedConstraint,load_data,metrics
DATA = Path(os.environ.get("NODE_LAC_DATA_DIR", str(Path(__file__).resolve().parents[1] / "results")))
from nodesac.core.manifold import FHNManifold
from nodesac.core.nodesac import ClosedLoopDynamics

class RevisionChecks(unittest.TestCase):
    def test_original_standardized_value_and_gradient(self):
        mean=torch.tensor([.2,.3,.5],dtype=torch.float64)
        std=torch.tensor([.4,.8,2.],dtype=torch.float64)
        base=FHNManifold(n_grid=1,e_threshold=1.)
        wrapped=StandardizedConstraint(base,mean,std)
        z=torch.tensor([[3.,2.,1.2]],dtype=torch.float64,requires_grad=True)
        torch.testing.assert_close(wrapped.k(z),base.k(z),rtol=0,atol=0)
        grad_wrapped=torch.autograd.grad(wrapped.distance(z).sum(),z)[0]
        grad_original=torch.autograd.grad(base.distance(z).sum(),z)[0]
        torch.testing.assert_close(grad_wrapped,grad_original,rtol=0,atol=0)
        x=mean+std*z
        torch.testing.assert_close((x[:,-1]-(mean[-1]+std[-1]))/std[-1],z[:,-1]-1.)

    def test_split_and_cached_values(self):
        if not all((DATA / (system + "_data.pt")).is_file() for system in runner.SYSTEMS):
            self.skipTest("Released datasets unavailable; set NODE_LAC_DATA_DIR to their directory")
        previous_data = runner.DATA
        runner.DATA = DATA
        self.addCleanup(setattr, runner, "DATA", previous_data)
        for system in ["fitzhugh_nagumo","lotka_volterra","shallow_water","franka_robot"]:
            tr,val,te,raw,times,mean,std,base,wrapped,p=load_data(system,"cpu")
            self.assertEqual(len(tr),102);self.assertEqual(len(val),26)
            self.assertFalse(set(p["fit_indices"]) & set(p["validation_indices"]))
            data=torch.load(DATA/(system+"_data.pt"),map_location="cpu",weights_only=False)
            torch.testing.assert_close(raw,data["test_states"],rtol=0,atol=0)
            torch.testing.assert_close(times,data["times"],rtol=0,atol=0)
            fit_raw=data["train_states"][p["fit_indices"]]
            torch.testing.assert_close(mean,fit_raw.reshape(-1,fit_raw.shape[-1]).mean(0))
            self.assertEqual(base.e_threshold,6.1 if system in ["fitzhugh_nagumo","shallow_water"] else 1.)
            torch.testing.assert_close(wrapped.k(te),base.k(te),rtol=0,atol=0)

    def test_ce_units_and_prediction_units(self):
        base=FHNManifold(n_grid=1,e_threshold=1.)
        z=torch.tensor([[[2.,1.,2.],[.1,.2,.3]]],dtype=torch.float64)
        true_z=torch.zeros_like(z)
        raw=10*z+100;true_raw=10*true_z+100
        result=metrics(raw,true_raw,base,z,true_z)
        self.assertAlmostEqual(result["CE"],float(base.k(z).square().sum(-1).mean()))
        self.assertAlmostEqual(result["MSE"],float((raw-true_raw).square().mean()))
        self.assertNotAlmostEqual(result["CE"],float(base.k(raw).square().sum(-1).mean()))

    def test_normalized_clipping_and_correction(self):
        mean=torch.zeros(3,dtype=torch.float64);std=torch.tensor([2.,.5,3.],dtype=torch.float64)
        wrapped=StandardizedConstraint(FHNManifold(n_grid=1,e_threshold=1.),mean,std)
        net=torch.nn.Linear(3,3,bias=False,dtype=torch.float64)
        torch.nn.init.zeros_(net.weight)
        gain=torch.nn.Sequential(torch.nn.Linear(3,1,dtype=torch.float64))
        torch.nn.init.zeros_(gain[0].weight);torch.nn.init.ones_(gain[0].bias)
        dyn=ClosedLoopDynamics(net,gain,wrapped,correction_scale=.3)
        z=torch.tensor([[2.,1.,2.]],dtype=torch.float64,requires_grad=True)
        k=wrapped.k(z);grad=torch.autograd.grad(k.square().sum(),z)[0]
        expected=-.3*torch.tanh(k.norm(dim=-1,keepdim=True))*grad.clamp(-5,5)
        torch.testing.assert_close(dyn(0.,z),expected)

    def test_dual_direction(self):
        for violation,sign in [(1.,1),(.0,-1)]:
            log_mu=torch.tensor(-2.3,dtype=torch.float64,requires_grad=True)
            before=float(log_mu.detach());opt=torch.optim.Adam([log_mu],lr=.01)
            loss=-log_mu*(violation-.01)
            opt.zero_grad();loss.backward();opt.step()
            self.assertGreater(sign*(float(log_mu.detach())-before),0.)

if __name__=="__main__":unittest.main()
