"""Evaluate the pinned official PhysicalRSI release and its full skill library."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from cryptography.fernet import Fernet

WORKSPACE = Path(__file__).absolute().parents[2]
INSTALL = Path('/data/ymz/physicalrsi-reproduction')
SIM = INSTALL / 'RoboDojo'
ADAPTER = SIM / 'XPolicyLab/policy/physicalRSI'
POLICY_PYTHON = INSTALL / 'skill-assets/implementations/openpi/.venv/bin/python'
SIM_PYTHON = Path('/home/ymz/miniconda3/envs/RoboDojo/bin/python')


def commit(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def stop(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait()


def descendants(pid):
    owned=[]
    try:
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return owned
    for child in children:
        child=int(child)
        owned.append(child)
        owned.extend(descendants(child))
    return owned


def stop_workers(owned):
    for pid in reversed(owned):
        try:
            if os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM)
            else:
                os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        live=[]
        for pid in owned:
            try:
                if Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z':live.append(pid)
            except FileNotFoundError:
                pass
        if not live:return
        time.sleep(.2)
    for pid in live:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--num-envs',type=int,default=1)
    parser.add_argument('--mode',choices=['full','align-code'],default='full')
    args = parser.parse_args()
    run_id = datetime.now().strftime(f'physicalrsi-official-{args.mode}-%Y%m%d-%H%M%S')
    out = WORKSPACE/'eval_result'/run_id
    out.mkdir(parents=True,exist_ok=False)
    provider = json.loads(Path('/home/ymz/.awomopaw.secret/providers/custom/gpt6.json').read_text())
    key = provider['api_key']
    if key.startswith('ENC:'):
        master = bytes.fromhex(Path('/home/ymz/.awomopaw.secret/.master_key').read_text().strip())
        key = Fernet(base64.urlsafe_b64encode(master)).decrypt(key[4:].encode()).decode()
    config = json.loads((INSTALL/'skills.full.json').read_text())
    config['agent'] = {'endpoint':provider['base_url'].rstrip('/')+'/chat/completions','model':'gpt-6-astra'}
    # Relocate only generated evidence; executable sources and dependency hashes stay pinned.
    for record in list(config['skills'].values())+list(config['program_registry'].values()):
        settings = record['configuration']
        if settings.get('runtime') == 'frozen-program-service':
            settings['output'] = str(out/'code-skills')
    if args.mode == 'align-code':
        # An explicit diagnostic of the user-selected skill. The default mode lets
        # the official agent choose freely from the complete registered library.
        config['compositions'] = ['memory-guided-program-library']
        config['program_registry'] = {'edge-alignment':config['program_registry']['edge-alignment']}
    config_path = out/'skills.json'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    deploy = out/'deploy.yml'
    deploy.write_text((ADAPTER/'deploy.yml').read_text())
    env = os.environ.copy()
    env.update(PHYSICALRSI_AGENT_API_KEY=key,PHYSICALRSI_SKILL_CONFIG=str(config_path),
        PHYSICALRSI_EVAL_OUTPUT=str(out/'harness'),XPL_DEPLOY_YML=str(deploy),
        PYTHONPATH=':'.join(map(str,[ADAPTER/'runtime',SIM,SIM/'XPolicyLab',
            SIM/'third_party/IsaacLab/source/isaaclab',
            SIM/'third_party/IsaacLab/source/isaaclab_assets',
            SIM/'third_party/IsaacLab/source/isaaclab_tasks',SIM/'third_party/curobo'])),
        PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.3',
        CUDA_VISIBLE_DEVICES='0',OMNI_KIT_ACCEPT_EULA='YES',EVAL_NUM=str(args.episodes),
        ROBODOJO_RUN_ID=run_id,PATH='/home/ymz/miniconda3/bin:'+env['PATH'])
    preflight = subprocess.run([str(POLICY_PYTHON),'-m',
        'PhysicalRSI_baselines.robodojo.skill_preflight',str(config_path)],
        env=env,capture_output=True,text=True)
    (out/'preflight.json').write_text(preflight.stdout)
    if preflight.returncode:
        print(preflight.stdout,preflight.stderr,flush=True)
        raise RuntimeError('Full skill preflight failed')
    metadata = dict(run_id=run_id,episodes_requested=args.episodes,seed=args.seed,
        num_envs=args.num_envs,mode=args.mode,
        physicalrsi_commit=commit(WORKSPACE/'PhysicalRSI'),xpolicylab_commit=commit(SIM/'XPolicyLab'),
        robodojo_commit=commit(SIM),isaaclab_commit=commit(SIM/'third_party/IsaacLab'),
        curobo_commit=commit(SIM/'third_party/curobo'),
        asset_manifest_sha256=hashlib.sha256((INSTALL/'assets.pinned.json').read_bytes()).hexdigest(),
        configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        installed_skills=sorted(config['skills']),operations=sorted(config['program_registry']),
        planner=config['agent'],policy_python=str(POLICY_PYTHON),sim_python=str(SIM_PYTHON),
        scope='Frozen official baseline in RoboDojo simulation; leaderboard identity is not assumed')
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    print('Run directory:',out,flush=True)
    server=client=None
    rc=1
    try:
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(['bash',str(ADAPTER/'setup_eval_policy_server.sh'),
                'RoboDojo','align_blocks','official-skill-library','arx_x5','joint',
                str(args.seed),'0',str(POLICY_PYTHON),str(port),'127.0.0.1'],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                if server.poll() is not None:
                    raise RuntimeError('Official server exited; inspect server.log')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Official server readiness timed out')
            print('Official policy server ready; starting simulator.',flush=True)
            with (out/'simulator.log').open('w') as log_client:
                client=subprocess.Popen([str(SIM_PYTHON),'-u','src/eval_client/main.py',
                    '--task_name','align_blocks','--env_cfg_type','arx_x5',
                    '--num_envs',str(args.num_envs),'--enable_cameras','--device_id','0',
                    '--policy_name','physicalRSI','--port',str(port),'--protocol','ws',
                    '--policy_server_url',f'ws://127.0.0.1:{port}',
                    '--additional_info','official-'+args.mode,'--seed',str(args.seed),
                    '--host','127.0.0.1','--headless'],cwd=SIM,env=env,
                    stdout=log_client,stderr=subprocess.STDOUT,start_new_session=True)
                rc=client.wait(timeout=max(1800,args.episodes*600))
    finally:
        workers=descendants(server.pid) if server is not None else []
        stop(client);stop(server);stop_workers(workers)
        for source in (SIM/'eval_result/RoboDojo/align_blocks/physicalRSI').glob(f'**/{run_id}'):
            shutil.copytree(source,out/'official',dirs_exist_ok=True)
        metadata['simulator_exit_code']=rc
        (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    result_path=out/'official/_result.json'
    if rc==0:
        if not result_path.is_file():
            raise RuntimeError('No official result; process completion is not task completion')
        result=json.loads(result_path.read_text())
        episodes=[]
        for p in out.glob('harness/*/episodes/*/episode.json'):
            e=json.loads(p.read_text())
            episodes.append({'episode':p.parent.name,'skill':e['skill'],'plan':e['plan']})
        summary={'official':result,'executed_episodes':episodes,'mode':args.mode}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if result.get('eval_time') != args.episodes:
            raise RuntimeError('Requested episode count was not completed')
        if len(episodes) != args.episodes:
            raise RuntimeError('Execution evidence missing for one or more episodes')
        print('Official score:',result['score'],'success rate:',result['success_rate'],flush=True)
        print('Executed skills:',[e['skill']['name'] for e in episodes],flush=True)
    print('Artifacts:',out,flush=True)
    raise SystemExit(rc)


if __name__=='__main__':
    main()
