"""Run official CodeProposer and SelfHarness with a native RoboDojo bridge.

Task strategy source is written by the configured LLM, never by this driver.
The released align_blocks skill and all other task policies are unavailable
to the generated worker. Generic segmentation, calibration and motion remain.
"""
import argparse
import ast
import base64
import copy
from datetime import datetime
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

from cryptography.fernet import Fernet

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
REPO = WORKSPACE / 'PhysicalRSI'
INSTALL = Path('/data/ymz/physicalrsi-reproduction')
SIM = INSTALL / 'RoboDojo'
XPL = SIM / 'XPolicyLab'
ASSETS = INSTALL / 'skill-assets/implementations/code-skills'
POLICY_PYTHON = INSTALL / 'skill-assets/implementations/openpi/.venv/bin/python'
SIM_PYTHON = Path('/home/ymz/miniconda3/envs/RoboDojo/bin/python')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WORKSPACE / 'runtime/physicalrsi-full'))
from run_align_blocks import stop, descendants, stop_workers
from PhysicalRSI_baselines.robodojo.proposal import CodeProposer
from PhysicalRSI_baselines.robodojo import skill_proposal
from PhysicalRSI_baselines.robodojo.rethinking import VideoRethinker, failed_episodes
from PhysicalRSI_core.self_harness import SelfHarness, now, selection
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json

TASK = 'align_blocks'
SCOPE = 'RoboDojo simulation: new LLM-generated policies using generic primitives; not historical skill-generation reproduction'
METHODS = {
    'scene()': 'Return objects block0/block1/block2 and set_square with position_m, yaw_rad, extent_xy_m and confidence; RGB estimates only. Object names are detected order, not hidden simulator handles. Approximate object plane z=.785, table=.765.',
    'reobserve()': 'Return state and initial_state. State has left/right_arm_joint_state, left/right_ee_joint_state, left/right_ee_pose=[x,y,z,w,x,y,z]. Coordinates are world meters, quaternion wxyz. Initial state is observed at episode start.',
    'approach(arm, position_m, quaternion_wxyz, steps=12)': 'Generic CuRobo collision-aware move to an end-effector world pose. May fail. steps controls sampled action count, not a duration.',
    'servo(arm, position_m, quaternion_wxyz=None, steps=12)': 'Generic inverse-kinematics movement with joint interpolation for a nearby Cartesian target. Defaults to observed EE orientation. No task sequence is built into it.',
    'move_joints(arm, joints, steps=12)': 'Generic joint interpolation. A complete episode allows at most 175 primitive action targets.',
    'gripper(arm, command, steps=3)': 'Normalized command: 1 open, 0 closed. Other joints remain at commanded positions.',
    'home(arm="left", steps=15)': 'Return to observed initial joint positions, with a short hold. Does not decide when the task is finished.',
    'hold(steps=3)': 'Hold current target joints and grippers.',
    'contact_to_wrist(position_m, quaternion_wxyz)': 'Convert gripper-tip contact position to wrist position by subtracting a 0.135m offset along the EE local x axis. Generic robot geometry.',
    'quaternion_from_euler(xyz_radians)': 'Generic Euler xyz to quaternion wxyz. Radians.',
}
API = {
    'schema': 'physicalrsi.local-native-primitive-api/v1',
    'task': TASK,
    'instruction': 'Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm.',
    'design_goal': 'Do not lift blocks. Complete alignment and return both arms to their initial joint configurations. Control budget is 200 requests; primitive movement budget is 175.',
    'methods': METHODS,
    'policy_signature': 'def policy(robot, memory)',
    'available_python': 'No imports. Basic Python builtins, conditions, arithmetic, lists, dictionaries, sorting and loops are available. Use only documented robot methods. Never access robot internals or attributes beginning with underscore.',
    'scope': 'No pretrained alignment policy, layout coordinates, simulator object poses or native success flags are exposed during execution.',
}


def admit_source(source):
    tree = skill_proposal.validate_policy_source(source)
    allowed = {name.split('(')[0] for name in METHODS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Generated policy cannot access private attributes')
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'robot':
            if node.attr not in allowed:
                raise ValueError('Undocumented robot attribute: ' + node.attr)
    return True


def manifest(root, name, **metadata):
    components = {kind: {kind + '.json': file_digest(root / (kind + '.json'))} for kind in CLOSURE}
    components['skills']['policy.py'] = file_digest(root / 'policy.py')
    return dict(id=name, root=str(root), components=components, **metadata)


class LLM:
    def __init__(self, provider_config=None, model=None):
        provider_path = Path(provider_config or '/home/ymz/.awomopaw.secret/providers/custom/gpt6.json')
        provider = json.loads(provider_path.read_text())
        self.endpoint = provider['base_url'].rstrip('/') + '/chat/completions'
        self.key = provider['api_key']
        if self.key.startswith('ENC:'):
            master = bytes.fromhex((provider_path.parents[2] / '.master_key').read_text().strip())
            self.key = Fernet(base64.urlsafe_b64encode(master)).decrypt(self.key[4:].encode()).decode()
        self.model = model or provider.get('model') or 'gpt-6-astra'
        self.output = None
        self.images = []
        self.config = SimpleNamespace(public=lambda: dict(endpoint=self.endpoint, model=self.model))

    def complete(self, messages, tools):
        messages = copy.deepcopy(messages)
        if self.images:
            messages[-1]['content'] = [dict(type='text', text=messages[-1]['content'])] + [
                dict(type='image_url', image_url={'url': 'data:' + (mimetypes.guess_type(p)[0] or 'image/png') + ';base64,' + base64.b64encode(p.read_bytes()).decode()})
                for p in self.images[:6]]
        payload = dict(model=self.model, messages=messages, temperature=0.2, max_tokens=12000)
        atomic_json(self.output / 'model-payload.json', payload)
        atomic_json(self.output / 'model-call-start.json', {'time': now(), 'image_files': list(map(str, self.images))})
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = json.loads(response.read())
        atomic_json(self.output / 'model-raw-response.json', raw)
        atomic_json(self.output / 'model-call-end.json', {'time': now(), 'usage': raw.get('usage')})
        message = raw['choices'][0]['message']
        return message['content'], message.get('tool_calls', []), raw.get('usage')


class NativeEvaluator:
    def __init__(self, root, episodes):
        self.root = root
        self.episodes = episodes
        self.round = 0
        self.cache = {}
        self.implementation = {str(p): file_digest(p) for p in [HERE / 'bridge.py', HERE / 'run_evolution.py',
            *HERE.joinpath('generic_tools').glob('*.py'), HERE / 'generic_tools/head_camera.json']}
        self.revision = digest(self.identity())

    def identity(self):
        return dict(kind='native-robodojo-generic-generated-policy/v1', implementation=self.implementation,
                    task=TASK, episodes=self.episodes, split='development seed0; selection seed1 disjoint batches; final seed2',
                    observation_boundary='RGB plus observed robot state', action_type='joint')

    def admit(self, candidate, output):
        try:
            admit_source((Path(candidate['root']) / 'policy.py').read_text())
            accepted, reason = True, 'Parsed code uses documented generic primitives'
        except Exception as error:
            accepted, reason = False, str(error)
        return dict(accepted=accepted, reason=reason, freeze_sha256=verify_harness(candidate),
                    evidence={'bridge_source_hashes': self.implementation})

    def layout_hashes(self, seed, ids):
        base = SIM / f'Assets/Eval_Layout/RoboDojo/arx_x5/{seed}'
        return [file_digest(base / f'align_blocks_{index}.json') for index in ids]

    def validation(self, comparison, output):
        ids = list(range((self.round - 1) * self.episodes, self.round * self.episodes))
        atomic_json(output / 'validation-cases.json', dict(seed=1, ids=ids))
        return dict(split='validation', comparison_sha256=digest(comparison), generated_at=now(),
                    layouts={TASK: self.layout_hashes(1, ids)},
                    admission_evidence={'validation-cases.json': file_digest(output / 'validation-cases.json')})

    def run_native(self, candidate, output, *, seed, ids, role):
        output.mkdir(parents=True, exist_ok=False)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        run_id = 'generated-' + output.name + '-' + datetime.now().strftime('%Y%m%d-%H%M%S')
        atomic_json(output / 'worker-config.json', dict(source=str(Path(candidate['root']) / 'policy.py'),
            memory=str(Path(candidate['root']) / 'memory_rules.json'), output=str(output / 'execution')))
        descriptor = json.loads((ASSETS / 'runtime/descriptors/align_blocks.json').read_text())
        dependency_paths = [str(ASSETS / p.replace('{assets}/', '')) for p in descriptor['pythonpath']
                            if p.startswith('{assets}/dependencies/')]
        pythonpath = [str(HERE), str(REPO), str(SIM), str(XPL), *dependency_paths,
            str(SIM / 'third_party/IsaacLab/source/isaaclab'),
            str(SIM / 'third_party/IsaacLab/source/isaaclab_assets'),
            str(SIM / 'third_party/IsaacLab/source/isaaclab_tasks')]
        env = os.environ.copy()
        env.update(PYTHONPATH=':'.join(pythonpath), PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
                   OMNI_KIT_ACCEPT_EULA='YES', ROBODOJO_RUN_ID=run_id, EVAL_NUM=str(len(ids)),
                   GENERATED_POLICY_CONFIG=str(output / 'worker-config.json'), GENERATED_POLICY_PORT=str(port),
                   GENERATED_LAYOUT_IDS=json.dumps(ids), PATH='/home/ymz/miniconda3/bin:' + env['PATH'])
        # The generated worker receives no model credentials or frozen skill registry.
        for key in list(env):
            if 'API_KEY' in key or key in ['PHYSICALRSI_SKILL_CONFIG']:
                env.pop(key, None)
        deploy = output / 'deploy.yml'
        deploy.write_text((XPL / 'policy/physicalRSI/deploy.yml').read_text())
        env['XPL_DEPLOY_YML'] = str(deploy)
        client_source = (SIM / 'src/eval_client/main.py').read_text()
        hook = '''\n# External experiment adapter: restrict cases without changing layout contents.\nfrom env.seed_manager.seed_manager import SeedManager\n_original_init_eval = SeedManager.init_eval\ndef _experiment_init_eval(self, *args, **kwargs):\n    _original_init_eval(self, *args, **kwargs)\n    allowed = json.loads(os.environ['GENERATED_LAYOUT_IDS'])\n    self.seed_list = [index for index in allowed if index in self.seed_list]\n    self.ed_idx = len(self.seed_list)\n    self.idx = 0\n    print('[Experiment] selected official layout ids:', self.seed_list, flush=True)\nSeedManager.init_eval = _experiment_init_eval\n'''
        marker = 'from src.eval_client.eval_env import create_eval_env'
        assert client_source.count(marker) == 1
        bootstrap = output / 'simulator-entry.py'
        bootstrap.write_text(client_source.replace(marker, marker + hook))
        atomic_json(output / 'evaluation-plan.json', dict(candidate_id=candidate['id'],
            freeze_sha256=verify_harness(candidate), seed=seed, ids=ids, role=role,
            case_hashes=self.layout_hashes(seed, ids), run_id=run_id,
            original_client_sha256=file_digest(SIM / 'src/eval_client/main.py'),
            experiment_client_sha256=file_digest(bootstrap), excluded_program='align-blocks',
            execution_source_sha256=file_digest(Path(candidate['root']) / 'policy.py')))
        print(f'Native {role}: {candidate["id"]}, seed={seed}, layouts={ids}, output={output}', flush=True)
        server = client = None
        try:
            with (output / 'server.log').open('w') as log:
                server = subprocess.Popen([str(POLICY_PYTHON), '-u', str(HERE / 'bridge.py')],
                    cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Generated policy server startup failed; see ' + str(output / 'server.log'))
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Generated server startup timeout')
            with (output / 'simulator.log').open('w') as log:
                client = subprocess.Popen([str(SIM_PYTHON), '-u', str(bootstrap), '--task_name', TASK,
                    '--env_cfg_type', 'arx_x5', '--num_envs', '1', '--enable_cameras', '--device_id', '0',
                    '--policy_name', 'physicalRSI', '--port', str(port), '--protocol', 'ws',
                    '--policy_server_url', f'ws://127.0.0.1:{port}', '--additional_info', 'generated-evolution',
                    '--seed', str(seed), '--host', '127.0.0.1', '--headless'],
                    cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                rc = client.wait(timeout=1200)
        finally:
            owned = descendants(server.pid) if server is not None else []
            stop(client)
            stop(server)
            stop_workers(owned)
            for source in (SIM / 'eval_result/RoboDojo/align_blocks/physicalRSI').glob(f'**/{run_id}'):
                shutil.copytree(source, output / 'official', dirs_exist_ok=True)
        result_path = output / 'official/_result.json'
        if rc != 0 or not result_path.exists():
            raise RuntimeError('Native evaluation did not complete; retain diagnostics at ' + str(output))
        result = read_json(result_path)
        if result['eval_time'] != len(ids):
            raise RuntimeError('Native evaluation coverage mismatch')
        print(f'Native outcome {candidate["id"]}: score={result["score"]}, SR={result["success_rate"]}', flush=True)
        return result

    def development(self, candidate, output):
        sha = verify_harness(candidate)
        if sha in self.cache:
            cached = self.cache[sha]
            atomic_json(output / 'development-reuse.json', {'source': str(cached['output']),
                'candidate_sha256': sha, 'reason': 'same frozen policy and fixed development cases'})
            return cached
        directory = output / 'development' / candidate['id']
        result = self.run_native(candidate, directory, seed=0, ids=list(range(self.episodes)), role='development')
        records = []
        images = []
        for episode in sorted(directory.glob('execution/episode-*')):
            events = [json.loads(line) for line in (episode / 'events.jsonl').read_text().splitlines()]
            # Deliver visible state, perception, commands and failures, never layout coordinates.
            records.append(dict(initial=events[:2], observations_and_actions=events[2:][-35:]))
            frames = sorted(episode.glob('head-*.png'))
            if frames:
                images += [frames[0], frames[-1]]
        feedback = dict(task=TASK, split='development', score=result['score'],
            success_rate=result['success_rate'], episodes=result['eval_time'], traces=records,
            note='Native outcome is terminal feedback; scene fields are RGB estimates, not ground truth.')
        atomic_json(directory / 'model-development-feedback.json', feedback)
        cached = dict(output=directory, result=result, feedback=feedback, images=images)
        self.cache[sha] = cached
        return cached

    def evaluate(self, candidate, comparison, cohort, output):
        spec = read_json(output / 'validation-cases.json')
        directory = output / 'validation-native' / candidate['id']
        result = self.run_native(candidate, directory, seed=spec['seed'], ids=spec['ids'], role='selection-validation')
        proofs = {str(p.relative_to(output)): file_digest(p) for p in [directory / 'official/_result.json',
                  directory / 'evaluation-plan.json', *directory.glob('execution/*/events.jsonl')]}
        details = {row['layout_id']: row for row in result['details'].values()}
        return dict(candidate_id=candidate['id'], kind='native_robodojo_evaluation', native_exit_code=0,
            comparison_sha256=digest(comparison), cohort_sha256=digest(cohort),
            freeze_sha256=verify_harness(candidate), evaluator_revision=self.revision,
            episodes=[dict(task=TASK, layout_sha256=sha, state='completed',
                score=details[index]['score'], success=details[index]['success'], evidence_sha256=proofs)
                for index, sha in zip(spec['ids'], cohort['layouts'][TASK])])


class Proposal:
    def __init__(self, evaluator, model, history):
        self.evaluator, self.model, self.history = evaluator, model, history
        self.proposer = CodeProposer(model)

    def identity(self):
        return dict(kind='official-CodeProposer-with-native-development/v1',
            implementation=file_digest(Path(__file__)), official=self.proposer.identity(),
            rethinking=VideoRethinker(self.model).identity())

    def develop(self, parent, output):
        data = self.evaluator.development(parent, output)
        self.rethink(parent, data)
        return dict(split='evolve', evidence={'feedback': file_digest(data['output'] / 'model-development-feedback.json')},
                    model_feedback=data['feedback'], costs={'development_episodes': data['result']['eval_time']})

    def rethink(self, candidate, data):
        if 'failure_rethinking' in data['feedback']:
            return
        source_root = Path(candidate['root'])
        reflections = []
        for episode in failed_episodes(data['result']):
            events_path = data['output'] / f'execution/episode-{int(episode):03d}/events.jsonl'
            initial = json.loads(events_path.read_text().splitlines()[0])
            plan = read_json(data['output'] / 'evaluation-plan.json')
            reflections.append(VideoRethinker(self.model).reflect(
                task=TASK, instruction=initial['instruction'], run_id=plan['run_id'],
                episode=episode, result=data['result'],
                videos=list(map(str, (data['output'] / 'official').glob('*.mp4'))),
                source=(source_root / 'policy.py').read_text(),
                memory=read_json(source_root / 'memory_rules.json'),
                output=data['output'] / 'rethinking' / f'episode-{int(episode):03d}'))
        data['feedback']['failure_rethinking'] = reflections
        atomic_json(data['output'] / 'model-development-feedback.json', data['feedback'])

    def propose(self, parent, feedback, output):
        generation = output / 'generation'
        generation.mkdir()
        self.model.output = generation
        self.model.images = self.evaluator.cache[verify_harness(parent)]['images']
        visible = [feedback['model_feedback']]
        if self.history:
            visible.append(dict(task=TASK, split='development', previous_generated_attempts=self.history[-5:],
                note='All outcomes below are development-only; no selection/final validation is supplied.'))
        source_root = Path(parent['root'])
        try:
            artifact = skill_proposal.propose(task=TASK, source=(source_root / 'policy.py').read_text(),
                memory=read_json(source_root / 'memory_rules.json'), feedback=visible, skill_api=API,
                proposer=self.proposer, output=generation)
            admit_source(artifact['source'])
        except Exception as error:
            atomic_json(generation / 'rejected-generation.json', {'error': f'{type(error).__name__}: {error}'})
            self.history.append(dict(error=f'{type(error).__name__}: {error}', raw_response_path=str(generation)))
            print('Model proposal rejected:', error, flush=True)
            return []
        candidate_root = output / 'candidate'
        shutil.copytree(source_root, candidate_root)
        (candidate_root / 'policy.py').write_text(artifact['source'] + '\n')
        atomic_json(candidate_root / 'skills.json', {'kind': 'LLM-generated-code-policy',
            'source': 'policy.py', 'source_sha256': file_digest(candidate_root / 'policy.py')})
        atomic_json(candidate_root / 'memory_rules.json', artifact['memory'])
        (generation / 'generated-policy.py').write_text(artifact['source'] + '\n')
        atomic_json(generation / 'generated-memory.json', artifact['memory'])
        child = manifest(candidate_root, f'candidate-r{self.evaluator.round:02d}',
            parent_sha256=verify_harness(parent), method='official_CodeProposer_model_written_revision',
            changes={'skills': 'policy.py', 'memory_rules': artifact['memory']},
            environment={'benchmark': 'RoboDojo', 'task': TASK},
            evidence={'proposal': file_digest(generation / 'skill-proposal.json')}, costs={'proposals': 1})
        developed = self.evaluator.development(child, output)
        self.rethink(child, developed)
        self.history.append(dict(source=artifact['source'], rationale=artifact['rationale'],
            memory=artifact['memory'], development_feedback=developed['feedback']))
        atomic_json(output / 'development-history.json', self.history)
        return [child]


class Selector:
    def identity(self):
        return {'source': file_digest(Path(selection.__file__))}

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    root = WORKSPACE / 'eval_result' / datetime.now().strftime('physicalrsi-self-evolution-%Y%m%d-%H%M%S')
    root.mkdir(parents=True)
    print('Evolution artifacts:', root, flush=True)
    model = LLM()
    evaluator = NativeEvaluator(root, args.episodes)
    history = []
    proposal = Proposal(evaluator, model, history)
    state = HarnessState(root / 'lineage')
    seed = root / 'bootstrap'
    seed.mkdir()
    for kind in CLOSURE:
        value = {'scope': SCOPE, 'kind': kind}
        if kind == 'foundation':
            value = model.config.public()
        elif kind == 'tools':
            value = API
        elif kind == 'dependencies':
            value = evaluator.implementation
        elif kind == 'memory_rules':
            value = {}
        atomic_json(seed / (kind + '.json'), value)
    (seed / 'policy.py').write_text('def policy(robot, memory):\n    robot.scene()\n    robot.hold(3)\n')
    seed_harness = manifest(seed, 'no-task-skill-bootstrap')
    atomic_json(root / 'experiment.json', dict(maximum_generation_rounds=args.rounds, development_episodes=args.episodes,
        selection_episodes_per_candidate=args.episodes, final_episodes=args.episodes, LLM=model.config.public(),
        removed_skill='align-blocks', removed_implementation='align_blocks_rgb_straightedge',
        available_execution_skills='only newly generated code plus generic RGB and motion primitives',
        official_proposer_source_sha256=file_digest(REPO / 'PhysicalRSI_baselines/robodojo/proposal.py'),
        official_loop_source_sha256=file_digest(REPO / 'PhysicalRSI_core/self_harness/__init__.py'),
        manual_task_policy_edits=0, scope=SCOPE, integration_sources=evaluator.implementation))
    rounds = []
    for index in range(1, args.rounds + 1):
        evaluator.round = index
        directory = root / f'round-{index:02d}'
        loop = SelfHarness(directory, state=state, proposer=proposal, evaluator=evaluator, selector=Selector(),
            scope=SCOPE, max_candidates=1,
            profile={'tasks': {TASK: {'weight': 1, 'episodes': args.episodes,
                'score_range': [0, 1], 'maximum_regression': 0}},
                'minimum_gain': 0, 'tie_tolerance': 1e-10},
            protocol={'identity': evaluator.revision, 'evaluation_kind': 'native_robodojo_evaluation'})
        if index == 1:
            state.initialize(seed_harness, policy=loop.config, scope=SCOPE)
        print(f'=== Official SelfHarness round {index}/{args.rounds} ===', flush=True)
        result = loop.run()
        selection_path = directory / 'selection.json'
        decision = read_json(selection_path)['output'] if selection_path.exists() else {'survivor_id': result['harness']['id']}
        rounds.append(dict(round=index, selected_harness=result['harness']['id'], decision=decision))
        atomic_json(root / 'rounds.json', rounds)
        print('Selected harness:', result['harness']['id'], flush=True)
        selected_result = directory / ('evaluate_' + result['harness']['id'] + '.json')
        if selected_result.exists():
            rows = read_json(selected_result)['output']['episodes']
            if all(row['success'] for row in rows):
                print('Selected policy succeeded on all selection cases; freezing before final test.', flush=True)
                break
    selected = state.resolve()['harness']
    final = evaluator.run_native(selected, root / 'final-test', seed=2,
        ids=list(range(args.episodes)), role='held-out-final-test')
    shutil.copy2(Path(selected['root']) / 'policy.py', root / 'selected-policy.py')
    atomic_json(root / 'summary.json', dict(rounds=rounds, selected_harness=selected['id'],
        selected_freeze_sha256=verify_harness(selected), final_test=final,
        manual_task_policy_edits=0, scope=SCOPE, artifacts=str(root)))
    print('Final held-out outcome:', final, flush=True)
    print('Saved all generation, native evaluation, selection and lineage records:', root, flush=True)


if __name__ == '__main__':
    main()
