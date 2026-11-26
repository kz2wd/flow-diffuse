"""Compute missing channel evaluation metrics; never write to DNS or model stores."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from .channel_eval_store import metrics_key, load_metrics, save_metrics
except ImportError:
    from channel_eval_store import metrics_key, load_metrics, save_metrics

HERE=Path(__file__).resolve().parent
DEFAULT_PROJECT='channel_full3_ptov'
DEFAULT_RUNS=('dainty-sunset-0','brave-forest-1')


def notebook_api():
    # Only execute function/definition cells; the interactive controls never run here.
    nb=json.loads((HERE/'channel_eval.ipynb').read_text()); ns={'__name__':'__channel_eval__'}
    for cell in nb['cells']:
        text=''.join(cell['source'])
        if cell['cell_type']!='code':continue
        if text.startswith('CHECKPOINT ='):
            text=text[text.index('def checkpoint_split('):text.index('\nresults = None')]
        elif not text.startswith(('from pathlib import Path','WALL_PRESSURE_TO_VELOCITY =','def fft_grid(','def plot_results(')):
            continue
        exec(compile(text,'channel_eval.ipynb','exec'),ns)
    exec('from train_channel import S3_MODEL_PREFIX',ns)
    return ns


def _match_grids(ns, val_root, va, train_axes):
    axes=ns['axes_for'](val_root,va)
    for ref,a in zip(train_axes,axes,strict=True):np.testing.assert_allclose(a,ref,rtol=1e-7,atol=1e-10)
    return axes


def _batches(zarr_array, indices, ns, ny,stride,selection,batch):
    # Read the next batch while the current batch is being evaluated.
    it=iter(ns['selections'](indices,batch))
    with ThreadPoolExecutor(max_workers=1) as pool:
        def fetch(s):return ns['read_batch'](zarr_array,s,ny,stride,selection=selection)
        work=pool.submit(fetch,next(it))
        for s in it:
            prior,work=work,pool.submit(fetch,s)
            yield prior.result()
        yield work.result()


def _spooled_metrics(ns, missing, va, selected, grid, masks, pdf_plane, do_pdf, device, batch, scratch, geometry, on_result):
    torch=ns['torch']; ny,stride,selection=geometry
    models={}; norms={}
    for name,rec in missing.items():
        checkpoint=rec['checkpoint']; cfg=checkpoint['config']
        torch_model=ns['make_model'](cfg['arch'],device).eval()
        torch_model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        models[name]=torch_model.to(memory_format=torch.channels_last_3d)
        norms[name]=tuple(torch.as_tensor(v,dtype=torch.float32,device=device) for v in rec['norm'])
    shape=(len(selected),selection['out_ch'],grid['nx'],ny,grid['nz'])
    # Temporarily retain paired physical fields on disk so exact PDF bins need no second inference or S3 pass.
    with tempfile.TemporaryDirectory(dir=scratch,prefix='channel-compare-') as tmp:
        maps={name:np.lib.format.open_memmap(Path(tmp)/f'{i}.npy',mode='w+',dtype='float32',shape=shape)
              for i,name in enumerate((None,*missing))}
        start=0; t0=time.monotonic()
        for x,target in _batches(va,selected,ns,ny,stride,selection,batch):
            end=start+len(x); maps[None][start:end]=target
            x=torch.from_numpy(x).to(device)
            with torch.inference_mode():
                for name,model in models.items():
                    xm,xs,ym,ys=norms[name]
                    pred=model(((x-xm)/xs).contiguous(memory_format=torch.channels_last_3d))*ys+ym
                    maps[name][start:end]=pred.cpu().numpy()
            if end==len(selected) or end//10>start//10 or start==0:
                print(f'Inference: {end}/{len(selected)} shared snapshots in {time.monotonic()-t0:.1f}s',flush=True)
            start=end
        if start!=len(selected):raise RuntimeError('Inference count mismatch')
        for m in maps.values():m.flush()
        for name in missing:
            def factory():
                for i in range(0,start,batch): yield maps[name][i:i+batch],maps[None][i:i+batch]
            print(f'Reducing {name} (profiles, spectra and PDFs={do_pdf})',flush=True)
            result=ns['evaluate'](factory,grid,masks,pdf_plane,do_pdf)
            if result['snapshots']!=start:raise RuntimeError('Reduction count mismatch')
            on_result(name,result)
        maps.clear()


def compare_channel_runs(runs=DEFAULT_RUNS, *, project=DEFAULT_PROJECT, root='prediction',
                         compute_missing=False, exact_pdfs=True, re_tau=200., batch_size=1,
                         scratch=None, device=None, api=None, indices=None):
    """Return verified Zarr metrics in input order. Only missing runs enter inference."""
    if not runs or len(set(runs))!=len(runs) or batch_size<1 or not np.isfinite(re_tau) or re_tau<=0:
        raise ValueError('Need unique runs, positive batch size, and positive Re_tau')
    ns=api if api is not None else notebook_api(); torch=ns['torch']; fs=ns['_s3fs']()
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if compute_missing and not fs.exists(root.split('/')[0]):raise FileNotFoundError(f'Metrics bucket does not exist: {root}')
    recs={}; shared=None; train_info={}
    for name in runs:
        # Validate path segments before use.
        metrics_key(root,project,name,{})
        ckpt=f'{ns["S3_MODEL_PREFIX"]}/{project}/{name}/ckpt_best.pt'
        raw=fs.cat_file(ckpt)
        checkpoint=torch.load(BytesIO(raw),map_location='cpu',weights_only=True); cfg=checkpoint['config']
        ds=cfg['data_selection']; vr,vsel=ns['checkpoint_split'](ds['validation'])
        if indices is not None:
            if not isinstance(indices,range) or indices.step<1 or not len(indices) or any(i not in vsel for i in indices):
                raise ValueError('Requested indices are not in this checkpoint validation split')
            vsel=indices
        if shared is None:
            va=ns['_open_data_zarr'](vr); axes=ns['axes_for'](vr,va)
        if vsel[-1]>=va.shape[0]:raise ValueError('Validation index exceeds source')
        ny,stride,selection=ns['spatial_selection'](cfg,va)
        identity=(ds['validation']['uri'],vsel,ny,stride,selection)
        if shared is None:shared=identity
        if shared!=identity:raise ValueError('Runs differ in validation source, snapshot selection, task, wall or sampled grid')
        meta=dict(schema='channel-eval-v1',checkpoint='s3://'+ckpt,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                  validation_uri=identity[0],validation_slice=[vsel.start,vsel.stop,vsel.step], snapshots=len(vsel),
                  train=ds['train'],endpoint=ns['S3_ENDPOINT'], exact_pdfs=bool(exact_pdfs),re_tau=float(re_tau),
                  selection=selection,stride=stride,target_depth=ny,
                  grid_sha256=[hashlib.sha256(a.astype('<f8').tobytes()).hexdigest() for a in axes])
        key=metrics_key(root,project,name,meta)
        recs[name]=dict(key=key,meta=meta,checkpoint=checkpoint)
    results={name:load_metrics(fs,r['key'],r['meta']) for name,r in recs.items()}
    missing={name:r for name,r in recs.items() if results[name] is None}
    for name in runs:print(f'{name}: '+('cache miss' if name in missing else 'loaded verified Zarr metrics'),flush=True)
    if not missing:return results
    if not compute_missing:raise FileNotFoundError('Missing metrics for '+', '.join(missing)+'; enable compute_missing to run inference')
    for name,r in missing.items():
        spec=r['meta']['train']; tr,selection_t=ns['checkpoint_split'](spec)
        if tr not in train_info:
            ta=ns['_open_data_zarr'](tr)
            train_info[tr]=(ta,ns['axes_for'](tr,ta))
        ta,train_axes=train_info[tr]
        if ta.shape[1:]!=va.shape[1:] or selection_t[-1]>=ta.shape[0]:raise ValueError('Training source shape mismatch')
        for a,b in zip(axes,train_axes,strict=True):np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-10)
        norm_meta=dict(algorithm='train_channel_global_float32_ddof1_1e-6_v1',uri=spec['uri'],slice=spec['slice'],
                       endpoint=ns['S3_ENDPOINT'],dtype=str(ta.dtype),shape=list(ta.shape[1:]),target_depth=ny,
                       stride=stride,n_out=selection['out_ch'],geometry=selection,
                       grid_sha256=[hashlib.sha256(a.astype('<f8').tobytes()).hexdigest() for a in train_axes])
        r['norm'],_=ns['cached_norm'](HERE/'.channel_norm_cache',norm_meta,False,None)
    x,full_y,z=axes
    for a in (x,z):np.testing.assert_allclose(np.diff(a),np.diff(a).mean(),rtol=1e-6)
    if any(len(a)%stride for a in (x,z)):raise ValueError('FFT stride must divide the periodic grid')
    lx,lz=len(x)*np.diff(x).mean(),len(z)*np.diff(z).mean()
    wall=selection['wall_index'] if selection['wall_index'] is not None else 0
    y=abs(full_y[:ny]-full_y[wall]); yp=y*re_tau/((full_y[-1]-full_y[0])/2)
    masks={name:(yp>=lo)&(yp<hi) for name,lo,hi in zip(('Viscous sublayer','Buffer layer','Nominal log/overlap','Outer/core'),(0,5,30,100),(5,30,100,np.inf),strict=True)}
    buf=np.flatnonzero(masks['Buffer layer']); pdf_plane=int(buf[np.argmin(abs(yp[buf]-15))]) if len(buf) else None
    grid=ns['fft_grid'](len(x[::stride]),len(z[::stride]),lx,lz)
    def publish(name,result):
        r=missing[name]; cfg=r['checkpoint']['config']
        result.update(y=y,y_plus=yp,k=grid['k'],k_mode_count=grid['count'],lx=lx,lz=lz,pdf_plane=pdf_plane,
                      validation_indices=np.asarray(vsel,dtype=int),component_labels=('u','v','w') if selection['out_ch']==3 else ('Q',),
                      checkpoint=r['meta']['checkpoint'],best_epoch_index=cfg.get('best_epoch'),layer_masks=masks)
        verified=save_metrics(fs,r['key'],r['meta'],result)
        np.testing.assert_array_equal(verified['validation_indices'],result['validation_indices'])
        np.testing.assert_array_equal(verified['mse_y'],result['mse_y'])
        results[name]=verified; print(f'Published and reopened: s3://{r["key"]}',flush=True)
    _spooled_metrics(ns,missing,va,vsel,grid,masks,pdf_plane,exact_pdfs,device,batch_size,scratch,(ny,stride,selection),publish)
    return results


def plot_comparison(results):
    import matplotlib.pyplot as plt
    ref=next(iter(results.values())); n=len(ref['component_labels']); layers=list(ref['spectra'])
    for m in results.values():
        for key in ('validation_indices','y_plus','k','k_mode_count','component_labels'): np.testing.assert_array_equal(ref[key],m[key])
    f,axs=plt.subplots(2,n,squeeze=False,figsize=(max(6,4*n),7),layout='constrained')
    f.suptitle(f'Channel comparison — {ref["snapshots"]} shared validation snapshots')
    for c,component in enumerate(ref['component_labels']):
        for i,(run,m) in enumerate(results.items()):
            axs[0,c].plot(m['y_plus'],m['mse_y'][c],label=run)
            xx=np.arange(len(layers)); w=.8/len(results)
            axs[1,c].bar(xx-.4+w*(i+.5),[m['spectra'][l]['paired'][c].sum() for l in layers],width=w,label=run)
        axs[0,c].set(title=component,xlabel='Nominal y⁺',ylabel='Physical MSE along y'); axs[0,c].grid(alpha=.25)
        axs[1,c].set(ylabel='Layer MSE (Fourier sum incl. DC)',xticks=xx,xticklabels=layers); axs[1,c].tick_params(axis='x',rotation=26)
    axs[0,0].legend()
    s,axs=plt.subplots(len(layers),2,squeeze=False,figsize=(12,2.65*len(layers)),layout='constrained')
    k=ref['k']; valid=(k>0)&(ref['k_mode_count']>0)
    for i,layer in enumerate(layers):
        for j,(field,title) in enumerate((('power_mse','Squared power-spectrum mismatch'),('paired','Paired Fourier MSE'))):
            ax=axs[i,j]
            for run,m in results.items(): ax.semilogy(k[valid],m['spectra'][layer][field].mean(axis=0)[valid],label=run)
            ax.set(xlabel='Physical |k|',ylabel=layer,title=title if i==0 else ''); ax.grid(alpha=.25)
            if i==0:ax.legend()
    return f,s


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',nargs='+',default=DEFAULT_RUNS);p.add_argument('--project',default=DEFAULT_PROJECT)
    p.add_argument('--compute-missing',action='store_true');p.add_argument('--device');p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',nargs=3,type=int);p.add_argument('--no-pdf',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    result=compare_channel_runs(args.runs,project=args.project,compute_missing=args.compute_missing,device=args.device,
                               batch_size=args.batch_size,indices=range(*args.indices) if args.indices else None,
                               scratch=args.output,exact_pdfs=not args.no_pdf)
    for fig,name in zip(plot_comparison(result),('mse_comparison.png','spectral_comparison.png'),strict=True):fig.savefig(args.output/name,dpi=155)
    summary={n:dict(snapshots=m['snapshots'],best_epoch=m['best_epoch_index'],mse=m['mse_global'],component_mse=m['mse_component'].tolist(),
                    layer_mse={k:float(v['paired'].sum(axis=-1).mean()) for k,v in m['spectra'].items()}) for n,m in result.items()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
