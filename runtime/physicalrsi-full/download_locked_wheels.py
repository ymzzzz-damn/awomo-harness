"""Fetch remaining frozen OpenPI wheels in ranges; retain lockfile identities."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote,urlparse
import urllib.request

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT=Path('/data/ymz/physicalrsi-reproduction')
spec=importlib.util.spec_from_file_location('ranges',Path(__file__).with_name('download_ranges.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.SEGMENTS=ROOT/'wheel-segments';r.SEGMENTS.mkdir(exist_ok=True)
WHEELS=ROOT/'locked-wheels';WHEELS.mkdir(exist_ok=True)


def main():
    log=(ROOT/'openpi-install.log').read_text()
    pending=set(re.findall(r'^Downloading (\S+)',log,re.M))-set(re.findall(r'^ Downloaded (\S+)',log,re.M))
    reqtext=(ROOT/'openpi.locked-requirements.txt').read_text().replace('\\\n',' ')
    versions={}
    for line in reqtext.splitlines():
        line=line.split('--hash=')[0].strip()
        if not line or line.startswith('#'):continue
        req=Requirement(line)
        if req.name in pending and (req.marker is None or req.marker.evaluate()):
            versions[req.name]=next(iter(req.specifier)).version
    lock=tomllib.loads((ROOT/'skill-assets/implementations/openpi/uv.lock').read_text())
    tag_order={tag:index for index,tag in enumerate(sys_tags())};tags=set(tag_order);parts=[]
    for name,version in versions.items():
        package=next(p for p in lock['package'] if p['name']==name and p['version']==version)
        candidates=[]
        for wheel in package['wheels']:
            filename=unquote(urlparse(wheel['url']).path.rsplit('/',1)[-1])
            matching=parse_wheel_filename(filename)[3]&tags
            if matching:candidates.append((min(tag_order[t] for t in matching),filename,wheel))
        if not candidates:raise RuntimeError(f'No locked compatible wheel for {name}')
        _,filename,wheel=min(candidates,key=lambda c:c[0])
        size=wheel.get('size')
        if size is None:
            with r.opener().open(urllib.request.Request(wheel['url'],headers={'Range':'bytes=0-0'}),timeout=45) as response:
                size=int(response.headers['Content-Range'].rsplit('/',1)[-1])
        parts.append(dict(name=filename,url=wheel['url'],size=size,
                          sha256=wheel['hash'].removeprefix('sha256:'),package=name,version=version))
    jobs=[]
    for part in parts:
        p=WHEELS/part['name']
        if p.exists() and p.stat().st_size==part['size'] and r.sha(p)==part['sha256']:continue
        jobs.extend((part,s,min(s+r.CHUNK-1,part['size']-1)) for s in range(0,part['size'],r.CHUNK))
    print('Frozen wheel packages:',versions,'ranges:',len(jobs),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        fs=[pool.submit(r.fetch,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            f.result()
            if i%10==0 or i==len(fs):print('Wheel ranges:',i,'/',len(fs),flush=True)
    for part in parts:
        p=WHEELS/part['name']
        if p.exists() and p.stat().st_size==part['size'] and r.sha(p)==part['sha256']:continue
        with p.open('wb') as out:
            for start in range(0,part['size'],r.CHUNK):
                end=min(start+r.CHUNK-1,part['size']-1)
                segment=r.SEGMENTS/f"{part['name']}.{start}-{end}"
                with segment.open('rb') as source:
                    for b in iter(lambda:source.read(8*1024*1024),b''):out.write(b)
        if p.stat().st_size!=part['size'] or r.sha(p)!=part['sha256']:
            raise RuntimeError('Frozen wheel hash mismatch: '+part['name'])
        print('Verified locked wheel:',part['name'],flush=True)
    (ROOT/'locked-wheels.receipt.json').write_text(json.dumps(parts,indent=2)+'\n')


if __name__=='__main__':main()
