#!/usr/bin/env python3
"""Independently verify full-sweep scene identities, official scores and videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--runner-scripts',type=Path)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    sys.path.insert(0,str(args.runner_scripts or Path(__file__).resolve().parent))
    import run_full_frozen_skills as runner
    root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    rows=json.loads((root/'results.json').read_text())
    attempts=json.loads((root/'attempts.json').read_text())
    cases=manifest['cases'];case_map={c['index']:c for c in cases}
    assert len(case_map)==len(cases)
    assert len({(c['task'],c['seed'],c['layout']) for c in cases})==len(cases)
    assert len({r['index'] for r in rows})==len(rows)
    assert {r['index'] for r in rows}<=set(case_map)
    providers={p['provider_id']:p for p in json.loads((root/'registry.snapshot.json').read_text())}
    video_count=0;verified=[]
    for row in rows:
        case=case_map[row['index']]
        for key in ['task','seed','layout','layout_sha256','provider_id','action_type']:
            assert row[key]==case[key],(case['case_id'],key)
        if not row.get('valid_episode'):continue
        dest=Path(row['output'])
        assert row['scene_identity_verified']
        selected=json.loads((dest/'selected-scene.json').read_text())
        assert selected['layout_sha256']==case['layout_sha256']
        assert selected['layout_file']==case['layout_file']
        assert sha(Path(case['layout_file']))==case['layout_sha256']
        official=json.loads((dest/'official-result.json').read_text())
        run=official['run'];result=run['result'];provider=providers[case['provider_id']]
        assert (run['task'],run['seed'],run['eval_num'])==(case['task'],case['seed'],1)
        assert run['provider_revision']==provider['revision']
        assert run['provider_checkpoint']==provider['checkpoint']
        assert result['eval_time']==1 and len(result['details'])==1
        detail=next(iter(result['details'].values()))
        assert detail['layout_id']==case['layout']
        assert abs(float(result['score'])-float(row['score']))<1e-9
        assert abs(100*float(detail['score'])-float(row['score']))<1e-9
        assert bool(official['task_succeeded'])==bool(row['success'])==bool(detail['success'])
        instruction=json.loads((dest/'initial-observation.json').read_text())['instruction']
        trace=[json.loads(line) for line in (dest/'harness-trace.jsonl').read_text().splitlines()]
        for event in trace:
            if 'official_instruction' in event:assert event['official_instruction']==instruction
            if event['event']=='action_chunk_ready':
                assert event['source']=='policy' and event['provider_id']==case['provider_id']
                assert event['subtask_goal']==event['optimized_task_prompt']==instruction
            if event['event']=='action_chunk_execution':
                assert event['source']=='policy' and event['provider_id']==case['provider_id']
        assert sum(e['executed'] for e in trace if e['event']=='action_chunk_execution')==row['actions']
        videos=json.loads((dest/'videos.json').read_text());assert len(videos)==3
        assert {Path(v['path']).name for v in videos}=={Path(v).name for v in official['videos']}
        for video in videos:
            path=Path(video['path'])
            assert sha(path)==video['sha256']==sha(Path(video['original']))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(path)],text=True,timeout=120))
            assert int(probe['streams'][0]['nb_read_frames'])>0
            video_count+=1
        verified.append(dict(task=row['task'],seed=row['seed'],layout=row['layout'],score=row['score'],
            success=row['success'],run_id=row['run_id'],output=row['output']))
    for path,want in manifest['source_hashes'].items():assert sha(Path(path))==want,path
    for case in cases:assert sha(Path(case['layout_file']))==case['layout_sha256']
    snapshot=args.runner_scripts or Path(__file__).resolve().parent
    for path,want in manifest['script_hashes'].items():assert sha(snapshot.parent/path)==want,path
    summary=runner.aggregate(cases,rows,attempts,json.loads((root/'summary.json').read_text())['status'])
    if not args.partial:
        assert len(rows)==len(cases),'Full sweep is not finished'
        assert summary['status'] in ['completed','completed_with_infrastructure_errors']
    summary.update(independent_verification=True,verified_videos=video_count,
        verified_source_files=len(manifest['source_hashes']),episodes=verified,
        verifier_sha256=sha(Path(__file__)))
    seed_scores=[]
    for task in dict.fromkeys(c['task'] for c in cases):
        for seed in sorted({c['seed'] for c in cases if c['task']==task}):
            requested=[c for c in cases if c['task']==task and c['seed']==seed]
            selected=[r for r in verified if r['task']==task and r['seed']==seed]
            seed_scores.append(dict(task=task,seed=seed,requested=len(requested),scored=len(selected),
                successes=sum(r['success'] for r in selected),
                mean_score=sum(r['score'] for r in selected)/len(selected) if selected else None))
    summary['by_task_seed']=seed_scores
    runner.base.save(root/('verified-progress.json' if args.partial else 'verified-summary.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes'},ensure_ascii=False))


if __name__=='__main__':main()
