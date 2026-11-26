"""Immutable, selection-aware Zarr cache for channel checkpoint diagnostics."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import zarr


def _json(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def metrics_key(root,project,run,meta):
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.','..') for x in (project,run)):
        raise ValueError('Invalid project or run identifier')
    digest=hashlib.sha256(_json(meta).encode()).hexdigest()
    return f'{root.rstrip("/")}/{project}/{run}/metrics/{digest}'


def _encode(group,value):
    if isinstance(value,np.ndarray):
        group.attrs['type']='array'
        if hasattr(group,'create_array'):
            group.create_array('values',data=value)
        else:
            group.array('values',value)
    elif isinstance(value,dict):
        group.attrs['type']='dict'; group.attrs['keys']=list(value)
        for k,v in value.items(): _encode(group.create_group(quote(k,safe='')),v)
    elif isinstance(value,(list,tuple)) and any(isinstance(x,(dict,list,tuple,np.ndarray)) for x in value):
        group.attrs['type']='list'; group.attrs['length']=len(value)
        for i,v in enumerate(value):_encode(group.create_group(str(i)),v)
    else:
        group.attrs['type']='value'
        group.attrs['value']=value.item() if isinstance(value,np.generic) else value


def _decode(group):
    t=group.attrs['type']
    if t=='array':return np.asarray(group['values'])
    if t=='dict':return {k:_decode(group[quote(k,safe='')]) for k in group.attrs['keys']}
    if t=='list':return [_decode(group[str(i)]) for i in range(group.attrs['length'])]
    if t=='value':return group.attrs['value']
    raise ValueError(f'Invalid stored type: {t}')


def _read_group(fs,key,info,meta):
    generation=info['generation']; files=info['files']
    if not re.fullmatch(r'[a-f0-9]{32}',generation) or '.zgroup' not in files:
        raise ValueError(f'Invalid cache marker: {key}')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for name,checksum in files.items():
            if name.startswith('/') or '..' in name.split('/'):
                raise ValueError(f'Unsafe metric key {name}')
            data=fs.cat_file(f'{key}/{generation}/metrics.zarr/{name}')
            if hashlib.sha256(data).hexdigest()!=checksum: raise ValueError(f'Bad metric checksum for {name}')
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        group=zarr.open_group(str(root),mode='r')
        if _json(group.attrs['provenance']) != _json(meta):raise ValueError(f'Cache provenance mismatch: {key}')
        return _decode(group)


def load_metrics(fs,key,meta):
    try: marker=fs.cat_file(f'{key}/_SUCCESS.json')
    except FileNotFoundError:return None
    return _read_group(fs,key,json.loads(marker),meta)


def missing_runs(fs,records):
    return [run for run,(key,meta) in records.items() if load_metrics(fs,key,meta) is None]


def save_metrics(fs,key,meta,result):
    marker=f'{key}/_SUCCESS.json'
    if fs.exists(marker): raise FileExistsError(marker)
    generation=uuid4().hex
    remote='s3' in fs.protocol
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'metrics.zarr'
        group=zarr.open_group(str(root),mode='w',zarr_format=2) if hasattr(zarr.Group,'create_array') else zarr.open_group(str(root),mode='w',zarr_version=2)
        group.attrs['provenance']=meta
        _encode(group,result)
        files={}
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            name=p.relative_to(root).as_posix(); data=p.read_bytes()
            files[name]=hashlib.sha256(data).hexdigest()
            dest=f'{key}/{generation}/metrics.zarr/{name}'
            if not remote:fs.makedirs(str(Path(dest).parent),exist_ok=True)
            fs.pipe_file(dest,data)
    info={'generation':generation,'files':files}
    _read_group(fs,key,info,meta)  # verify before making this generation visible
    data=_json(info).encode()
    if fs.protocol=='s3' or 's3' in fs.protocol:
        bucket, obj, _=fs.split_path(marker)
        fs.call_s3('put_object',Bucket=bucket,Key=obj,Body=data,IfNoneMatch='*')
    else:
        fs.makedirs(key,exist_ok=True)
        with fs.open(marker,'xb') as f:f.write(data)
    return load_metrics(fs,key,meta)
