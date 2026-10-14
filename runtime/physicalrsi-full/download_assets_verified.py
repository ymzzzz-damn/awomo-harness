"""Resume official release parts, verify them, then run the shipped installer."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/ymz/physicalrsi-reproduction')
MANIFEST = ROOT / 'assets.pinned.json'
ASSETS = ROOT / 'skill-assets'
CACHE = ASSETS / '.downloads'
ADAPTER = Path(__file__).absolute().parents[2] / 'PhysicalRSI/PhysicalRSI_baselines/robodojo/XPolicyLab/policy/physicalRSI'
PYTHON = '/home/ymz/miniconda3/envs/RoboDojo/bin/python'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def fetch(part):
    target = CACHE / part['name']
    if target.exists() and target.stat().st_size == part['size'] and sha(target) == part['sha256']:
        return part['name']
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(30):
        if not (partial.exists() and partial.stat().st_size == part['size']):
            result = subprocess.run(['curl','--fail','--location','--silent','--show-error',
                '--proxy','http://127.0.0.1:7890','--retry','0',
                '--connect-timeout','30','--speed-limit','1024','--speed-time','90',
                '--continue-at','-','--output',str(partial),part['url']])
            if result.returncode:
                print('Resuming after transport interruption:',part['name'],
                      partial.stat().st_size if partial.exists() else 0,flush=True)
                time.sleep(min(10,attempt+1))
                continue
        if partial.stat().st_size == part['size'] and sha(partial) == part['sha256']:
            partial.replace(target)
            return part['name']
        partial.unlink()
    raise RuntimeError('Asset integrity failure: '+part['name'])


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    order = ['openpi-inference-source','physicalrsi-code-skills','pi05-weights','sparse-memory-weights']
    assets = sorted(manifest['assets'],key=lambda a: order.index(a['name']))
    parts = [p for a in assets for p in a['parts']]
    completed=[]
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures={executor.submit(fetch,p):p for p in parts}
        for future in concurrent.futures.as_completed(futures):
            name=future.result();completed.append(name)
            receipt={'manifest_sha256':sha(MANIFEST),'verified_parts':sorted(completed),
                     'total_parts':len(parts),'elapsed_s':round(time.monotonic()-start,1)}
            (ROOT/'download-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(f'Verified {len(completed)}/{len(parts)}: {name}',flush=True)
    for asset in assets:
        destination = ASSETS / asset['destination']
        if destination.exists():
            print('Installed asset already exists:',destination,flush=True)
            continue
        subprocess.run([PYTHON,str(ADAPTER/'download_assets.py'),'--manifest',str(MANIFEST),
            '--output',str(ASSETS),'--asset',asset['name']],check=True)
    print('All official assets downloaded, verified and installed.',flush=True)


if __name__ == '__main__':
    main()
