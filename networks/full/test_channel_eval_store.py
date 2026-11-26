import tempfile
import unittest
from pathlib import Path
import numpy as np
import fsspec
from networks.full.channel_eval_store import metrics_key, load_metrics, save_metrics, missing_runs

class MetricsStoreTest(unittest.TestCase):
    def test_round_trip_and_skip_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fs = fsspec.filesystem('file'); meta={'checkpoint_sha256':'a'*64,'indices':[200,201]}
            key=metrics_key(tmp,'project','first',meta); second=metrics_key(tmp,'project','second',meta)
            self.assertIn('/project/first/metrics/',key)
            self.assertIsNone(load_metrics(fs,key,meta))
            numeric={'mse':np.array([2.,3.]),'layers':{'outer/core':np.ones((2,2))},'pdf':[{},{'edges':np.array([0,1])}],'count':2,'none':None}
            save_metrics(fs,key,meta,numeric)
            out=load_metrics(fs,key,meta)
            self.assertIsNone(out['none']); self.assertEqual(out['count'],2)
            np.testing.assert_array_equal(out['layers']['outer/core'],np.ones((2,2)))
            np.testing.assert_array_equal(out['pdf'][1]['edges'],[0,1])
            self.assertEqual(missing_runs(fs,{'first':(key,meta),'second':(second,meta)}),['second'])
            self.assertNotEqual(key,metrics_key(tmp,'project','first',{**meta,'indices':[200]}))
            with self.assertRaises(FileExistsError):save_metrics(fs,key,meta,numeric)
    def test_incomplete_tree_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            key=metrics_key(tmp,'p','r',{}); Path(key,'orphan').mkdir(parents=True)
            self.assertIsNone(load_metrics(fsspec.filesystem('file'),key,{}))

if __name__=='__main__':unittest.main()
