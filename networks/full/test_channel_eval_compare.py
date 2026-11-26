import tempfile
import unittest
from pathlib import Path
import json

import fsspec
import numpy as np
import torch
import zarr

from networks.full.channel_eval_compare import compare_channel_runs, notebook_api, plot_comparison


class ChannelComparisonTest(unittest.TestCase):
    def test_cached_runs_do_not_repeat_inference_when_a_run_is_added(self):
        torch.set_num_threads(1)
        ns=notebook_api(); ns['S3_MODEL_PREFIX']='models'
        fs=fsspec.filesystem('file'); ns['_s3fs']=lambda:fs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'prediction').mkdir(); ns['S3_MODEL_PREFIX']=str(root/'models')
            ds=zarr.open_array(str(root/'val.zarr'),mode='w',shape=(5,5,8,7,6),chunks=(1,5,8,7,6),dtype='float32')
            fields=np.random.default_rng(4).normal(size=ds.shape).astype('float32');ds[:]=fields
            axes=[np.arange(8)*.5,np.linspace(0,2,7),np.arange(6)*.25]
            ns['axes_for']=lambda n,a:axes
            reads=[]
            class Counted:
                def __getitem__(self,s):reads.extend(range(*s[0].indices(ds.shape[0]))); return ds[s]
                def __getattr__(self,k):return getattr(ds,k)
            ns['_open_data_zarr']=lambda n:Counted() if n=='val' else ds
            selection=ns['WALL_PRESSURE_TO_VELOCITY']; stats=ns['training_norm'](ds,range(3),4,1,2,selection=selection)
            ns['cached_norm']=lambda *args:(stats,root/'unused')
            cfg=dict(arch=dict(c=4,n_blocks=1,dy=4,out_ch=3),task='wall_pressure_to_velocity',wall='lower',half_y=True,xz_stride=1,
                     data_selection={'train':dict(uri='s3://simulations/train/data.zarr',slice=[0,3,1],snapshots=3),
                                     'validation':dict(uri='s3://simulations/val/data.zarr',slice=[2,5,1],snapshots=3)},best_epoch=2)
            for label in ('first','second','third'):
                where=root/'models'/'p'/label; where.mkdir(parents=True)
                model=ns['make_model'](cfg['arch'],'cpu'); torch.save({'model_state_dict':model.state_dict(),'config':cfg},where/'ckpt_best.pt')
            args=dict(project='p',root=str(root/'prediction'),compute_missing=True,batch_size=2,scratch=root,device='cpu',api=ns)
            initial=compare_channel_runs(['first','second'],**args); self.assertEqual(reads,[2,3,4]); reads.clear()
            again=compare_channel_runs(['first','second','third'],**args); self.assertEqual(reads,[2,3,4]); reads.clear()
            cached=compare_channel_runs(['first','second','third'],**{**args,'compute_missing':False}); self.assertEqual(reads,[])
            for name in initial:
                np.testing.assert_array_equal(initial[name]['mse_y'],cached[name]['mse_y'])
            for name,m in cached.items():
                np.testing.assert_array_equal(m['validation_indices'],[2,3,4]); self.assertEqual(m['snapshots'],3)
                cfg0=torch.load(root/'models'/'p'/name/'ckpt_best.pt',weights_only=True)
                model=ns['make_model'](cfg0['config']['arch'],'cpu').eval();model.load_state_dict(cfg0['model_state_dict'])
                pairs=list(ns['prediction_batches'](ds,range(2,5),4,1,2,model,stats,'cpu'))
                p,t=(np.concatenate([b[i] for b in pairs]) for i in (0,1))
                np.testing.assert_allclose(m['mse_y'],np.mean((p-t)**2,axis=(0,2,4)),rtol=2e-6)
                self.assertEqual(m['pdf'][0][0]['counts'].sum(),2*3*8*4*6)
            for fig in plot_comparison(again):fig.canvas.draw()
            ns['plt'].close('all')

if __name__=='__main__':unittest.main()
