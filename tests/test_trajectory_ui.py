"""Exercise evidence adapters and the HTTP video contract without robot/model calls."""
import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trajectory_ui import EvidenceStore, Handler, classify, read_events


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_outcomes_never_count_unscored_or_infrastructure_as_task_failure():
    assert classify({'valid_episode': False, 'success': False, 'status': 'native_unstable_before_actions'}, {}) == 'unscored'
    assert classify({'valid_episode': False, 'success': False, 'status': 'infrastructure_error'}, {}) == 'error'
    assert classify({'status': 'running'}, {}) == 'running'
    assert classify({'status': 'pending'}, {}) == 'pending'
    assert classify({'status': 'completed'}, {'eval_time': 1, 'details': {'0': {'success': False}}}) == 'failure'
    assert classify({'success': False}, {'eval_time': 1, 'details': {'0': {'success': True}}}) == 'success'


def test_inventory_keeps_pending_and_scene_logs_have_real_actions(tmp_path):
    batch = tmp_path / 'batch'
    episode = batch / 'episode-00-seed-0-layout-0'
    save(batch / 'experiment.json', {'task': 'align_blocks'})
    save(batch / 'official-case-inventory.json', [{'seed': 0, 'layout_id': 0}, {'seed': 0, 'layout_id': 1}])
    log = tmp_path / 'sim.log'
    log.write_text('ALIGN_registered {"count":1,"stage":"release_and_home","action":{"left_arm_joint_state":[0.12]}}\n')
    save(episode / 'official-result.json', {'run': {'run_id': 'one', 'task': 'align_blocks', 'seed': 0, 'log_path': str(log),
        'status': 'completed', 'result': {'eval_time': 1, 'details': {'0': {'layout_id': 0, 'success': True}}}}})
    (episode / 'harness-trace.jsonl').write_text('{"event":"observation","official_instruction":"original"}\n{"unfinished":')
    store = EvidenceStore(tmp_path)
    scenes = store.scan()['scenes']
    assert len(scenes) == 2
    assert sorted(s['status'] for s in scenes) == ['pending', 'success']
    detail = store.detail(next(s['id'] for s in scenes if s['status'] == 'success'))
    assert detail['instruction'] == 'original'
    assert detail['steps'][0]['action']['left_arm_joint_state'] == [0.12]
    assert detail['total_steps'] == 1
    assert detail['steps'][0]['line'] == 1


def test_chat_official_record_wins_over_internal_worker_receipts(tmp_path):
    batch = tmp_path / 'chat-batch'
    scene = batch / 'scene-01-layout-3'
    episode = scene / 'run-one'
    save(scene / 'receipt.json', {'runs': [{'run_id': 'one'}]})
    save(episode / 'code-policies/code-a/receipt.json', {'run_id': 'one', 'policy_id': 'code-a', 'status': 'returned'})
    save(scene / 'visible-messages.json', [{'role': 'assistant', 'text': 'Saved decision'}])
    save(scene / 'model-tool-calls.json', [{'name': 'start_session', 'input': {'policy_id': 'pi05'}}])
    save(episode / 'official-result.json', {'run': {'run_id': 'one', 'task': 'align_blocks', 'seed': 1, 'status': 'completed',
        'result': {'eval_time': 1, 'details': {'0': {'layout_id': 3, 'success': False}}}}})
    store = EvidenceStore(tmp_path)
    scenes = store.scan()['scenes']
    assert len(scenes) == 1
    assert scenes[0]['status'] == 'failure'
    assert scenes[0]['layout'] == 3
    detail = store.detail(scenes[0]['id'])
    assert detail['messages'][0]['text'] == 'Saved decision'
    assert len(detail['calls']) == 1


def test_refresh_replaces_pending_with_completed(tmp_path):
    batch = tmp_path / 'batch'
    save(batch / 'experiment.json', {'task': 'align_blocks'})
    save(batch / 'official-case-inventory.json', [{'seed': 0, 'layout_id': 0}])
    store = EvidenceStore(tmp_path)
    assert store.scan()['scenes'][0]['status'] == 'pending'
    save(batch / 'episode/receipt.json', {'task': 'align_blocks', 'run_id': 'done', 'seed': 0, 'layout_id': 0,
                                        'valid_episode': True, 'success': True, 'status': 'completed'})
    scenes = store.scan(force=True)['scenes']
    assert len(scenes) == 1 and scenes[0]['status'] == 'success'


def test_http_range_seek_and_unknown_paths(tmp_path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'0123456789')
    store = EvidenceStore(tmp_path)
    url = store.file_url(video)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.store = store
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(Request(base + url, headers={'Range': 'bytes=2-5'})) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == 'bytes 2-5/10'
            assert response.read() == b'2345'
        with urlopen(Request(base + url, headers={'Range': 'bytes=-3'})) as response:
            assert response.read() == b'789'
        for path, code, headers in [(url, 416, {'Range': 'bytes=20-'}), ('/media/../../etc/passwd', 404, {})]:
            try:
                urlopen(Request(base + path, headers=headers))
                assert False, 'Invalid request unexpectedly succeeded'
            except HTTPError as error:
                assert error.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_full_scene_manifest_preserves_layouts_running_and_pending(tmp_path):
    batch = tmp_path / 'full'
    save(batch / 'manifest.json', {'schema': 'awomopaw.full-official-scenes/v1', 'cases': [
        {'task': 'push_T', 'seed': 0, 'layout': layout, 'provider_id': 'frozen-push'} for layout in range(3)]})
    save(batch / 'results.json', [{'task': 'push_T', 'run_id': 'one', 'seed': 0, 'layout': 2,
                                 'valid_episode': True, 'success': True, 'status': 'completed'}])
    save(batch / 'status.json', {'running': [{'task': 'push_T', 'seed': 0, 'layout': 1}]})
    scenes = EvidenceStore(tmp_path).scan()['scenes']
    assert {s['layout']: s['status'] for s in scenes} == {0: 'pending', 1: 'running', 2: 'success'}
    assert next(s['provider'] for s in scenes if s['layout'] == 0) == 'frozen-push'


def test_user_cancelled_native_score_does_not_count_as_failure(tmp_path):
    folder = tmp_path / 'chat/stopped-by-user/run-one'
    save(folder / 'run-state.json', {'task': 'align_blocks', 'run_id': 'one', 'seed': 0, 'status': 'completed'})
    save(folder / 'native-finalization.json', {'run': {'result': {'eval_time': 1, 'score': 0, 'details': {'0': {'success': False}}}}})
    scenes = EvidenceStore(tmp_path).scan()['scenes']
    assert len(scenes) == 1
    assert scenes[0]['status'] == 'unscored'
    assert scenes[0]['score'] is None


def test_ablation_has_two_complete_inventories_and_fresh_agent_evidence(tmp_path):
    batch = tmp_path / 'paired'
    save(batch/'manifest.json', {'task':'align_blocks', 'seed':0,
        'arms':['without_skills','with_skills'],
        'cases':[{'layout_id':i} for i in range(55)]})
    episode = batch/'layout-00/with_skills'
    save(episode/'receipt.json', {'arm':'with_skills','layout_id':0,'seed':0,
        'run_id':'one','status':'completed','valid_episode':True,'score':100,'success':True})
    (episode/'agent-messages.jsonl').write_text(json.dumps({'message':{'role':'assistant','content':[
        {'type':'text','text':'Read original skill and plan the push.'},
        {'type':'tool_use','name':'robodojo_sim__request_policy_chunk','input':{'provider_id':'pi05'}}]}})+'\n')
    store=EvidenceStore(tmp_path)
    catalog=store.scan()
    assert {e['id']:e['scenes'] for e in catalog['experiments']} == {
        'paired::with_skills':55,'paired::without_skills':55}
    assert len(catalog['scenes']) == 110
    scene=next(s for s in catalog['scenes'] if s['arm']=='with_skills' and s['layout']==0)
    detail=store.detail(scene['id'])
    assert detail['messages'][0]['text']=='Read original skill and plan the push.'
    assert detail['calls'][0]['input']['provider_id']=='pi05'
    pending=next(s for s in catalog['scenes'] if s['arm']=='without_skills' and s['layout']==0)
    save(batch/'layout-00/without_skills/receipt.json', {'arm':'without_skills','layout_id':0,
         'seed':0,'run_id':'two','status':'running'})
    refreshed=store.scan(force=True)
    assert len(refreshed['scenes']) == 110
    assert next(s for s in refreshed['scenes'] if s['id']==pending['id'])['status']=='running'

    save(batch/'layout-00/without_skills/initial-observation.json', {'instruction':'Official live goal'})
    assert store.detail(pending['id'])['instruction']=='Official live goal'


def test_online_code_evolution_links_executed_versions_without_counting_final_update(tmp_path):
    catalog_root, artifacts = tmp_path / 'catalog', tmp_path / 'external-artifacts'
    batch = catalog_root / 'online'
    save(batch / 'experiment.json', {'schema': 'physicalrsi.online-code-evolution/v1',
        'artifact_root': str(artifacts), 'task': 'align_blocks', 'title': '空记忆逐场自进化',
        'provider': 'GPT-6-Astra · Code-policy', 'qualification': '最终 policy 未单独评测'})
    save(artifacts / 'summary.json', {'task': 'align_blocks', 'seed': 0, 'post_run_policy_evaluated': False})
    save(artifacts / 'error.json', {'error': 'Historical startup failure, now resolved'})
    save(artifacts / 'official/_result.json', {'eval_time': 2, 'details': {
        '0': {'layout_id': 11, 'success': False, 'score': 0},
        '1': {'layout_id': 54, 'success': True, 'score': 1}}})
    save(artifacts / 'case-results.json', [{'case': 0}, {'case': 1}])
    for index in range(3):  # The last model output was generated after the two evaluations.
        case = f'case-{index:03d}'
        save(artifacts / 'updates' / case / 'skill-proposal.json', {'rationale': f'Repair {index}'})
        save(artifacts / 'candidates' / case / 'memory_rules.json', {'lesson': f'v{index}'})
        (artifacts / 'candidates' / case / 'policy.py').write_text(f'# policy {index}\n')
        save(artifacts / 'ipc/assignments' / f'{case}.json', {'candidate': {'id': f'case-policy-{index:03d}'}})
    save(artifacts / 'ipc/requests/case-001.json', {'instruction': 'Align and reset.'})
    save(artifacts / 'feedback/case-001.json', {'native_outcome': {'success': True}})
    trace = artifacts / 'execution/episode-001/events.jsonl'
    trace.parent.mkdir(parents=True)
    trace.write_text('\n'.join(json.dumps(e) for e in [
        {'event': 'initial', 'step': 0},
        {'event': 'servo', 'step': 0, 'arm': 'right', 'pose': [0.1, 0.2, 0.3, 1, 0, 0, 0]},
        {'event': 'move_joints', 'step': 5, 'arm': 'right', 'joints': [0.25]},
        {'event': 'policy_returned', 'step': 7}]))
    (artifacts / 'official/episode_0000001_cam_head_success.mp4').write_bytes(b'video')
    artifact = artifacts / 'task-artifact'
    artifact.mkdir()
    (artifact / 'SKILL.md').write_text('New skill')
    (artifact / 'policy.py').write_text('# Untested final candidate\n')
    store = EvidenceStore(catalog_root)
    catalog = store.scan()
    assert catalog['experiments'][0]['title'] == '空记忆逐场自进化'
    assert len(catalog['scenes']) == 2
    assert sorted(s['status'] for s in catalog['scenes']) == ['failure', 'success']
    scene = next(s for s in catalog['scenes'] if s['layout'] == 54)
    assert scene['policy_id'] == 'case-policy-001'  # Episode index differs from layout ID.
    assert scene['cameras']['cam_head']
    detail = store.detail(scene['id'])
    assert detail['error'] is None
    assert detail['instruction'] == 'Align and reset.'
    assert detail['steps'][0]['action'] == {'right_arm_joint_state': [0.25]}
    assert detail['steps'][0]['line'] == 3
    assert detail['total_steps'] == 7
    assert detail['evidence'][0]['target'] == [0.1, 0.2, 0.3]
    assert detail['learning']['source_layouts'] == [11]
    assert detail['learning']['update_rationale'] == 'Repair 2'
    assert detail['messages'] == [{'role': 'assistant', 'text': 'Repair 1'}]
    files = {f['label']: Path(f['path']) for f in detail['files']}
    assert files['本场景实际执行的 policy'] == artifacts / 'candidates/case-001/policy.py'
    assert files['最终蒸馏的 policy · 未单独评测'] == artifact / 'policy.py'
    assert files['最终蒸馏的 SKILL'] == artifact / 'SKILL.md'


def test_discovered_policy_and_skill_are_served_as_readable_text(tmp_path):
    store = EvidenceStore(tmp_path)
    documents = {'policy.py': '# Saved source, never executed\n', 'SKILL.md': '# 蒸馏技能\n'}
    urls = {}
    for name, content in documents.items():
        path = tmp_path / name
        path.write_text(content)
        urls[name] = store.file_url(path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.store = store
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name, url in urls.items():
            with urlopen(f'http://127.0.0.1:{server.server_port}{url}') as response:
                assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
                assert response.read().decode() == documents[name]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_paired_online_ui_uses_real_candidate_path_and_original_selection(tmp_path):
    root=tmp_path/'catalog';batch=root/'rsi';artifacts=tmp_path/'run'
    candidate=artifacts/'round-001/candidate';candidate.mkdir(parents=True)
    (candidate/'policy.py').write_text('def policy(robot, memory):\n    robot.hold(1)\n')
    save(candidate/'memory_rules.json',{'lesson':'new'})
    save(candidate/'manifest.json',{'id':'candidate-r01'})
    save(batch/'experiment.json',{'schema':'physicalrsi.online-code-evolution/v1',
        'artifact_root':str(artifacts),'task':'align_blocks','cases':[0,1],
        'paired_validation':True,'framework':'RSI','video_rethink_enabled':True})
    save(artifacts/'official/_result.json',{'eval_time':1,'details':{'1':{'layout_id':1,'success':True,'score':1}}})
    save(artifacts/'ipc/assignments/case-001.json',{'candidate':{'id':'candidate-r01','root':str(candidate)}})
    save(artifacts/'updates/case-001/accepted-proposal.json',{'directory':str(artifacts/'round-001/generation')})
    save(artifacts/'round-001/generation/skill-proposal.json',{'rationale':'Generated from scene0'})
    old=artifacts/'round-001/validation-native/old'
    save(old/'official/_result.json',{'eval_time':1,'details':{'0':{'layout_id':1,'success':True,'score':1}}})
    (old/'official/episode_0000000_cam_head_success.mp4').write_bytes(b'video')
    save(artifacts/'selections/case-001.json',{'incumbent_before':'old','candidate':'candidate-r01',
        'incumbent_after':'old','decision':{'decision':'retain_parent'},'incumbent_directory':str(old),
        'source_scene':0,'validation_scene':1})
    store=EvidenceStore(root);scenes=store.scan()['scenes']
    scene=next(s for s in scenes if s['layout']==1)
    assert scene['paired_validation'] is True
    detail=store.detail(scene['id'])
    assert detail['comparison']['candidate_score']==detail['comparison']['incumbent_score']==1
    assert detail['comparison']['decision']=='retain_parent'
    assert detail['comparison']['survivor']=='old'
    assert detail['learning']['task_memory']=='{\n  "lesson": "new"\n}'
    assert any(f['path']==str(candidate/'policy.py') for f in detail['files'])
    assert any(f['label'].startswith('对照旧版视频') for f in detail['files'])
    bootstrap=store.detail(next(s['id'] for s in scenes if s['layout']==0))
    assert bootstrap['learning']['video_rethinking']['status']=='bootstrap'
