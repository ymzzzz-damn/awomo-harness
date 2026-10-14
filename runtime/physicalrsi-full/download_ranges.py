"""Finish official, uncompressed release parts with resumable HTTP ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path('/data/ymz/physicalrsi-reproduction')
CACHE=ROOT/'skill-assets/.downloads'
SEGMENTS=ROOT/'download-segments'
CHUNK=32*1024*1024


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))


def fetch(job):
    part,start,end=job
    p=SEGMENTS/f"{part['name']}.{start}-{end}"
    expected=end-start+1
    for attempt in range(30):
        got=p.stat().st_size if p.exists() else 0
        if got==expected:return p
        if got>expected:raise RuntimeError('Unexpected segment size')
        request=urllib.request.Request(part['url'],headers={'Range':f'bytes={start+got}-{end}'})
        try:
            with opener().open(request,timeout=45) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f"bytes {start+got}-{end}/{part['size']}":
                    raise RuntimeError('Server did not honor the exact byte range')
                with p.open('ab') as f:
                    remaining=expected-got
                    while remaining:
                        b=r.read(min(1024*1024,remaining))
                        if not b:raise EOFError('Interrupted segment')
                        f.write(b);f.flush();remaining-=len(b)
        except Exception as e:
            if attempt==29:raise
            time.sleep(min(attempt+1,10))
    raise RuntimeError('Range download exhausted retries')


def main():
    SEGMENTS.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'assets.pinned.json').read_text())
    assets=sorted(manifest['assets'],key=lambda a:0 if a['name']=='physicalrsi-code-skills' else 1)
    for asset in assets:
        jobs=[];plans=[]
        for part in asset['parts']:
            target=CACHE/part['name']
            if target.exists() and target.stat().st_size==part['size'] and sha(target)==part['sha256']:
                continue
            partial=target.with_suffix(target.suffix+'.partial')
            start=partial.stat().st_size if partial.exists() else 0
            if start>part['size']:raise RuntimeError('Oversized partial')
            chunks=[(part,s,min(s+CHUNK-1,part['size']-1)) for s in range(start,part['size'],CHUNK)]
            jobs.extend(chunks);plans.append((part,partial,chunks))
        print(asset['name'],'missing ranges:',len(jobs),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            futures=[pool.submit(fetch,job) for job in jobs]
            for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                future.result()
                if i%10==0 or i==len(futures):print(asset['name'],i,'/',len(futures),'ranges complete',flush=True)
        for part,partial,chunks in plans:
            with partial.open('ab') as out:
                for p,start,end in chunks:
                    segment=SEGMENTS/f"{p['name']}.{start}-{end}"
                    with segment.open('rb') as source:
                        for b in iter(lambda:source.read(8*1024*1024),b''):out.write(b)
            if partial.stat().st_size!=part['size'] or sha(partial)!=part['sha256']:
                raise RuntimeError('Official part checksum mismatch: '+part['name'])
            partial.replace(CACHE/part['name'])
            print('Verified',part['name'],flush=True)
            for p,start,end in chunks:(SEGMENTS/f"{p['name']}.{start}-{end}").unlink()
        receipt={'verified_parts':[p['name'] for a in manifest['assets'] for p in a['parts']
            if (CACHE/p['name']).exists()], 'total_parts':sum(len(a['parts']) for a in manifest['assets']),
            'manifest_sha256':sha(ROOT/'assets.pinned.json')}
        (ROOT/'download-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('All official release parts verified.',flush=True)


if __name__=='__main__':main()
