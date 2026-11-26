"""Run the two full_3 trials sequentially, then compare all four checkpoints."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4


CONFIGS = ({'c':128,'n_blocks':5}, {'c':128,'n_blocks':10})
BASELINES = ('dainty-sunset-0','brave-forest-1')
PROJECT = 'channel_full3_ptov'


def install_activation_checkpointing():
    import torch
    from torch.utils.checkpoint import checkpoint
    from networks.full.channel_model import DataNet
    def forward(self, p):
        x = self.core[0](p.expand(-1,-1,-1,self.dy,-1))
        for j in range(1,len(self.core)-1):
            layer = self.core[j]
            x = checkpoint(layer,x,use_reentrant=False,preserve_rng_state=False) if self.training else layer(x)
        return self.core[-1](x)
    DataNet.forward = forward


def gpu_probe(torch):
    from time import monotonic
    from networks.full.channel_model import make_model
    for arch in CONFIGS:
        model=make_model({**arch,'dy':65,'out_ch':3},'cuda').train()
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        x=torch.randn((4,1,128,1,128),device='cuda'); y=torch.randn((4,3,128,65,128),device='cuda')
        for step in range(2):
            t=monotonic(); torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True); torch.nn.functional.mse_loss(model(x),y).backward(); opt.step()
            torch.cuda.synchronize()
            print(f'Synthetic GPU step {arch} #{step+1}: {monotonic()-t:.2f}s; peak reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB',flush=True)
        del opt,model,x,y
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/workspace/projects'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check',action='store_true',help='Check CUDA and print the plan, without opening any data')
    p.add_argument('--probe',action='store_true',help='Exercise both full-grid architectures on synthetic CUDA data only')
    p.add_argument('--workers',type=int,help='Override DataLoader worker count (use 0 for small /dev/shm)')
    args=p.parse_args()
    repo=args.repo.resolve(); out=args.output.resolve()
    if args.workers is not None and args.workers<0:p.error('--workers must be nonnegative')
    if not (repo/'networks/full/channel_eval_compare.py').is_file():p.error('Missing project comparison module')
    os.chdir(repo); sys.path.insert(0,str(repo))
    import torch
    usable=torch.cuda.is_available()
    print(json.dumps(dict(project=PROJECT,architectures=CONFIGS,epochs=20,batch_size=4,
                          train=[200,3196,1],validation=[200,1001,1],cuda_available=usable)),flush=True)
    if not usable:
        raise SystemExit('CUDA not visible to this Python process. No dataset was loaded and no training was started.')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}',flush=True)
    if args.check:return
    install_activation_checkpointing()
    if args.probe:
        gpu_probe(torch)
        return
    out.mkdir(parents=True,exist_ok=False)
    import trackio
    from networks.full import train_channel as tr
    from networks.full.channel_eval_compare import compare_channel_runs, plot_comparison
    if args.workers is not None: tr.NUM_WORKERS=args.workers
    data=tr.load_training_data(task='wall_pressure_to_velocity',max_samples=5000,half_y=True,xz_stride=1,log=True)
    if (data.selection['train']['slice'],data.selection['validation']['slice'])!=([200,3196,1],[200,1001,1]):
        raise RuntimeError('Full_3 splits changed; do not compare with old runs silently')
    fs=tr._s3fs(); original_init=trackio.init; records=[]
    for arch in CONFIGS:
        name=f"full3-c{arch['c']}-b{arch['n_blocks']}-{uuid4().hex[:10]}"
        expected=f'{tr.S3_MODEL_PREFIX}/{PROJECT}/{name}/ckpt_best.pt'
        if fs.exists(expected):raise FileExistsError(expected)
        with patch.object(trackio,'init',lambda **kw:original_init(name=name,**kw)):
            res=tr.run_trial(arch,data=data,device='cuda',epochs=20,lr=1e-3,trackio_project=PROJECT,
                             trial_label=name,spectral_image_samples=2,spectral_image_every=1,log=True)
        if res['checkpoint_uri']!='s3://'+expected or res['interrupted_checkpoint_uri'] is not None:
            raise RuntimeError(f'No complete verified tracked run for {name}')
        ck=torch.load(BytesIO(fs.cat_file(expected)),map_location='cpu',weights_only=True)
        assert ck['config']['arch']==res['arch'] and ck['config']['data_selection']==data.selection
        records.append(dict(run=name,checkpoint=res['checkpoint_uri'],best_epoch=res['best_epoch'],arch=res['arch']))
        (out/'trained_runs.json').write_text(json.dumps(records,indent=2)+'\n')
        del res,ck
        torch.cuda.empty_cache()
    del data
    results=compare_channel_runs([*BASELINES,*(r['run'] for r in records)],project=PROJECT,
                                 compute_missing=True,device='cuda',scratch=out)
    for fig,name in zip(plot_comparison(results),('mse_comparison.png','spectral_comparison.png'),strict=True):
        fig.savefig(out/name,dpi=155)
    summary={name:dict(snapshots=m['snapshots'],mse=m['mse_global'],component_mse=m['mse_component'].tolist()) for name,m in results.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
