#!/usr/bin/env python3
"""Isolation, inherited-configuration, and frozen-validation replay checks."""
import json,math,sys,time,unittest,os
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tune_validation as tuning

class TuningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.info = None
        if os.environ.get("NODE_LAC_ARCHIVE_TESTS") == "1":
            try:
                cls.info = tuning.prepared()
            except (FileNotFoundError, PermissionError):
                pass
    def setUp(self):
        if self._testMethodName in {
                "test_cache_isolation_and_values", "test_default_validation_replay"}:
            if self.info is None:
                self.skipTest("Optional historical archive check; set NODE_LAC_ARCHIVE_TESTS=1 with its inputs")
            if self._testMethodName == "test_default_validation_replay" and torch.cuda.device_count() < 2:
                self.skipTest("Historical checkpoint replay requires CUDA device 1")
    def test_cache_isolation_and_values(self):
        original_load=torch.load
        allowed={tuning.CACHE/f"{s}.pt" for s in tuning.frozen.SYSTEMS}
        def guarded(path,*args,**kwargs):
            if Path(path) not in allowed:raise AssertionError("Loader opened a non-cache artifact")
            return original_load(path,*args,**kwargs)
        with patch.object(torch,"load",side_effect=guarded),patch.object(tuning.frozen,"load_data",side_effect=AssertionError("Forbidden frozen loader")):
            for system in tuning.frozen.SYSTEMS:
                payload=original_load(tuning.CACHE/f"{system}.pt",map_location="cpu",weights_only=True)
                self.assertFalse(set(payload)-{"train_states","times","k_max"})
                self.assertEqual(tuning.tensor_sha(payload["train_states"]),self.info["cache"][system]["train_tensor_sha256"])
                self.assertEqual(tuning.tensor_sha(payload["times"]),self.info["cache"][system]["times_tensor_sha256"])
                tr,val,times,constraint,provenance=tuning.load_train_validation(system,"cpu",self.info)
                self.assertEqual(len(tr),102);self.assertEqual(len(val),26);self.assertEqual(len(times),100)
                self.assertFalse(set(provenance["fit_indices"])&set(provenance["validation_indices"]))
                tuning.verify_preprocessing(provenance,self.info["reuse"][f"{system}/0/42"]["preprocessing"])
    def test_exact_training_dispatch(self):
        sentinels=[object() for _ in range(4)]
        for index in range(4):
            with patch.object(tuning.frozen,"train_lac",return_value="called") as called:
                out=tuning.train_new(*sentinels,"cuda:1",Path("/unused"),index)
                self.assertEqual(out,"called")
                called.assert_called_once_with(*sentinels,"cuda:1",100,Path("/unused"),tuning.options(index))
        tuning.check_frozen()
    def test_grid_and_selection_ties(self):
        self.assertEqual(tuning.RHOS,[0,.05,.1,.15,.2,.25,.3,.4,.5,.75,1,1.5,2])
        rows=[{"rho":rho,"finite_mse":True,"validation":{"MSE":1.}} for rho in tuning.RHOS]
        self.assertEqual(tuning.choose(rows,tuning.RHOS)["rho"],0)
        rows[0]["finite_mse"]=False
        self.assertEqual(tuning.choose(rows,tuning.RHOS)["rho"],.05)
        if self.info is not None:
            self.assertEqual(len(self.info["reuse"]),21)
    def test_default_validation_replay(self):
        comparisons=[]
        for system in tuning.frozen.SYSTEMS:
            tr,val,times,constraint,provenance=tuning.load_train_validation(system,"cuda:1",self.info)
            for seed in tuning.SEEDS:
                inherited=self.info["reuse"][f"{system}/0/{seed}"]
                path=Path(inherited["checkpoint_path"])
                self.assertEqual(tuning.sha(path),inherited["checkpoint_sha256"])
                node,gain,_=tuning.model_from_checkpoint(path,tr.shape[-1],constraint,"cuda:1",0)
                rows=tuning.score_rhos(node,gain,val,times,constraint,tuning.COARSE)
                expected={float(k):v for k,v in inherited["metadata"]["selection"]["validation_scale_scores"].items()}
                differences=[]
                for row in rows:
                    if row["rho"] not in expected:
                        self.assertFalse(row["finite_mse"]);continue
                    actual=row["validation"]["MSE"];reference=expected[row["rho"]]
                    self.assertTrue(math.isclose(actual,reference,rel_tol=1e-10,abs_tol=1e-12),(system,seed,row["rho"],actual,reference))
                    differences.append(abs(actual-reference))
                self.assertEqual(tuning.choose(rows,tuning.COARSE)["rho"],inherited["metadata"]["selection"]["scale"])
                comparisons.append({"system":system,"seed":seed,"maximum_absolute_difference":max(differences)})
        self.assertEqual(len(comparisons), 12)

if __name__ == "__main__":
    unittest.main()
