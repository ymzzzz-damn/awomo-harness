#!/usr/bin/env python3
"""Run the real AwomoPaw QwenPawAgent, with fresh memory and restricted tools.

Uses the existing default model configuration without updating it. Only the
selected skill reference files can be read; no shell or general filesystem tools.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from run_parallel_frozen_skills import save, now

COMMON = '''You are the AwomoPaw planner for a RoboDojo simulation evaluation.
Use the exact official instruction, current public RGB cameras and proprioception.
Only pi05 may generate actions. Never run source code, direct actions, frozen
controllers, other providers or another episode. All motion goes through the
robodojo_sim embodiment. Start align_blocks seed 0 once, wait for readiness,
including status initializing, which is normal while Pi0.5 and Isaac start.
Poll session_status until it returns a current observation; startup may take
several minutes. Do not abort while initializing or treat startup as a model
failure. Only explicit failed lifecycle state or an expired startup budget
indicates a startup error. Before readiness do not request camera frames.
After readiness read current_observation and capture all three cameras. Preserve every official
constraint in optimized_task_prompt. Plan one current subtask, request_policy_chunk
with the fresh observation_id and pi05, execute its unchanged prefix with
max_actions=10 and discard_remaining=true, then observe all three cameras and
replan. Continue until the episode ends, then read evaluation_result. Read no
reward, object ground truth or official layout files during execution. On model
or simulator errors abort and report the error. Do not stop merely because task
progress is poor; use the official action budget. Describe visible evidence in
planning_summary. If a task skill is registered, read its SKILL.md and original
controller reference before the first action, and use it only for semantic
planning. Distinguish source-document instructions from this execution contract.
'''
PROMPT = '完成 RoboDojo 的 align_blocks，seed 0。由 awomopaw 规划，只用 pi05 执行动作；按评测控制协议运行到 episode 结束，保存官方结果。'
ALLOWED = {'health', 'start_session', 'session_status', 'current_observation',
    'capture_head_frame', 'capture_left_wrist_frame', 'capture_right_wrist_frame',
    'request_policy_chunk', 'execute_action_chunk', 'abort_session',
    'evaluation_status', 'evaluation_result'}


async def run(args):
    from agentscope.message import Msg, TextBlock
    from agentscope.tool import ToolResponse
    from qwenpaw.agents.react_agent import QwenPawAgent
    from qwenpaw.app.embodiments.manager import EmbodimentManager
    from qwenpaw.agents.context import LightContextManager
    from qwenpaw.config.config import AgentProfileConfig, ToolsConfig

    dest = args.output
    workspace = dest / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    # Both arms have the same common prompt, tool permissions and budgets.
    (workspace / 'AGENTS.md').write_text(COMMON)
    config = AgentProfileConfig.model_validate(json.loads(args.agent_config.read_text()))
    config.workspace_dir = str(workspace)
    config.running.max_iters = 1200
    config.running.auto_continue_on_text_only = True
    config.running.memory_manager_backend = 'none'
    config.tools = ToolsConfig()
    for tool in config.tools.builtin_tools.values():
        tool.enabled = False
    embodiment = Path(__file__).resolve().parents[1] / 'embodiments/robodojo_sim'
    if not embodiment.is_dir():
        raise RuntimeError('Pinned ablation embodiment package is missing')
    for package in config.embodiments.packages.values():
        if package.package_id != 'robodojo_sim':
            package.enabled = False
            continue
        package.path = str(embodiment)
        package.bindings['robodojo_sim_api'] = args.bridge
    manager = EmbodimentManager()
    await manager.init_from_config(config.embodiments, workspace)
    context = LightContextManager(working_dir=str(workspace), agent_id='default')
    await context.start()
    if manager.get_active_tool_count() < 10:
        raise RuntimeError('Embodiment tools did not load')
    definitions = manager.get_loader().get_tools()
    execute = next(t['function'] for t in definitions if t['function']['name'].endswith('.execute_action_chunk'))
    if not {'max_actions', 'discard_remaining'}.issubset(execute['parameters']['properties']):
        raise RuntimeError('Embodiment schema does not expose short execution windows')

    class RecordedAgent(QwenPawAgent):
        async def print(self, msg, last=True, speech=None):
            if last:
                with (dest / 'agent-messages.jsonl').open('a') as stream:
                    stream.write(json.dumps({'at': now(), 'message': msg.to_dict()}, ensure_ascii=False) + '\n')
                save(dest / 'agent-state.json', self.state_dict())
            # No modification of model content or action selection.

    agent = RecordedAgent(config, embodiment_manager=manager, context_manager=context, workspace_dir=workspace,
        request_context={'agent_id': 'default', 'channel': 'console', 'session_id': dest.name,
                         'user_id': 'pi05-skills-evaluation'})
    # Remove implicit orchestration and all non-benchmark tools.
    for name in list(agent.toolkit.tools):
        if name not in {'robodojo_sim__' + op for op in ALLOWED}:
            agent.toolkit.remove_tool_function(name)
    agent.toolkit.skills.clear()
    permitted = {}
    reads = []
    if args.arm == 'with_skills':
        agent.toolkit.register_agent_skill(str(args.skill))
        permitted = {p.resolve(): p for p in args.skill.rglob('*') if p.is_file()}

    async def read_file(file_path: str, start_line: int = 1, end_line: int = 100000) -> ToolResponse:
        """Read an installed task skill reference. Only registered skill files are available.

        Args:
            file_path: Full path shown in the registered skill or its references.
            start_line: First line, inclusive, one-based.
            end_line: Last line, inclusive.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        if path not in permitted:
            return ToolResponse(content=[TextBlock(type='text', text='File unavailable in this experimental arm.')])
        start_line, end_line = int(start_line), int(end_line)
        if start_line < 1 or end_line < start_line:
            raise ValueError('Invalid line interval')
        lines = path.read_text().splitlines()
        text = '\n'.join(f'{i+1}: {line}' for i, line in enumerate(lines) if start_line <= i+1 <= end_line)
        reads.append({'path': str(path), 'start_line': start_line, 'end_line': min(end_line, len(lines)), 'at': now()})
        save(dest / 'skill-reads.json', reads)
        return ToolResponse(content=[TextBlock(type='text', text=text)])

    agent.toolkit.register_tool_function(read_file)
    save(dest / 'agent-runtime.json', {'agent_class': 'qwenpaw.agents.react_agent.QwenPawAgent',
        'model': config.active_model.model, 'model_provider': config.active_model.provider_id,
        'fresh_memory': True, 'memory_backend': 'none', 'arm': args.arm,
        'context_backend': 'light',
        'tools': list(agent.toolkit.tools), 'skills': list(agent.toolkit.skills),
        'execute_schema': execute,
        'common_prompt': COMMON, 'user_prompt': PROMPT})
    (dest / 'system-prompt.txt').write_text(agent.sys_prompt)
    if args.dry_run:
        save(dest / 'agent-finished.json', {'status': 'dry_run', 'no_episode_started': True})
        await manager.close()
        await context.close()
        return
    started = time.monotonic()
    try:
        reply = await asyncio.wait_for(agent(Msg('user', PROMPT, 'user')), args.timeout)
        save(dest / 'agent-reply.json', reply.to_dict())
        save(dest / 'agent-state.json', agent.state_dict())
        save(dest / 'agent-finished.json', {'status': 'returned', 'seconds': time.monotonic()-started})
    except BaseException as error:
        save(dest / 'agent-state.json', agent.state_dict())
        save(dest / 'agent-error.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        await manager.close()
        await context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--bridge', required=True)
    parser.add_argument('--arm', choices=['with_skills', 'without_skills'], required=True)
    parser.add_argument('--skill', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=5400)
    parser.add_argument('--dry-run', action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
