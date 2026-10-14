"""Distill completed development evidence into a frozen task artifact candidate.

Candidates are skill text, executable code and memory, not model weight updates.
Independent Self-Harness validation/selection remains required for promotion.
"""
import copy
import ast
import json
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, identifier
from .rethinking import VideoRethinker, _json_call, episode_outcomes
from .skill_proposal import validate_policy_source


def verify_artifact(artifact, task):
    if (artifact.get('schema') != 'awomopaw.task-artifact/v1' or artifact.get('task') != task
            or artifact.get('revision') != digest(artifact.get('content'))):
        raise ValueError('Task artifact identity mismatch')
    content = artifact['content']
    if (set(content) != {'skill_text', 'source', 'memory'}
            or not isinstance(content['skill_text'], str)
            or not isinstance(content['memory'], dict) or not isinstance(content['source'], str)):
        raise ValueError('Invalid task artifact content')
    if content['source']:
        tree = validate_policy_source(content['source'])
        if not any(isinstance(node, ast.FunctionDef) and node.name == 'policy'
                   and [arg.arg for arg in node.args.args] == ['robot', 'memory']
                   for node in tree.body):
            raise ValueError('Task policy must define policy(robot, memory)')
    return copy.deepcopy(content)


def evolve_task(*, model, final, instruction, parent, output, primitive_api=None,
                reflections=None, executed_content=None, split='development', samples=8):
    # Reject inappropriate feedback before accessing videos or making model calls.
    run = final['run']
    if split != 'development' or run['status'] != 'completed':
        raise ValueError('Task evolution accepts completed development episodes only')
    task = identifier(run['task'])
    outcomes = episode_outcomes(run['result'])
    if not instruction or not isinstance(instruction, str):
        raise ValueError('The original public instruction is required')
    if any(row.get('run_id') != run['run_id'] for row in (reflections or [])):
        raise ValueError('Chunk reflections must belong to this completed run')
    content = verify_artifact(parent, task) if parent else dict(skill_text='', source='', memory={})
    executed = executed_content if executed_content is not None else content
    if primitive_api and primitive_api.get('task') not in (None, task):
        raise ValueError('Primitive API task mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    video_feedback = [VideoRethinker(model, samples=samples, include_success=True).reflect(
        task=task, instruction=instruction, run_id=run['run_id'], episode=episode,
        result=run['result'], videos=final['videos'], source=executed['source'],
        memory={**executed['memory'], 'current_task_skill': executed['skill_text']},
        output=output / 'videos' / f'episode-{int(episode):03d}') for episode in outcomes]
    feedback = dict(task=task, split=split, instruction=instruction,
                    native_result=run['result'], video_feedback=video_feedback,
                    chunk_reflections=reflections or [])
    atomic_json(output / 'development-feedback.json', feedback)
    candidate = _json_call(model,
        'Distill this task\'s completed development feedback into a reusable task skill and policy. '
        'Use the exact original instruction. Preserve successful mechanisms and repair observed '
        'failures. Distinguish hypotheses from observed facts and retain uncertainty. '
        'Do not memorize scene coordinates, seed/layout answers, hidden state or scoring shortcuts. '
        'Return ONLY JSON: skill_text (nonempty Markdown string), source (Python defining '
        'policy(robot, memory), or empty string when no primitive API is supplied), memory '
        '(JSON object), rationale (nonempty string), validation_plan (nonempty list of strings). '
        'Code may call only the supplied primitive API. This is a candidate; independent '
        'evaluation must precede promotion and no success-rate gain is established by reflection.',
        json.dumps(dict(parent=content, executed=executed, feedback=feedback, primitive_api=primitive_api), ensure_ascii=False),
        output / 'distillation-call')
    if (not isinstance(candidate, dict)
            or set(candidate) != {'skill_text', 'source', 'memory', 'rationale', 'validation_plan'}
            or not isinstance(candidate['skill_text'], str) or not candidate['skill_text'].strip()
            or not isinstance(candidate['rationale'], str) or not candidate['rationale'].strip()
            or not isinstance(candidate['validation_plan'], list) or not candidate['validation_plan']
            or any(not isinstance(x, str) or not x.strip() for x in candidate['validation_plan'])
            or (not primitive_api and candidate['source'])):
        raise ValueError('Invalid task distillation candidate')
    revised = {key: candidate[key] for key in ('skill_text', 'source', 'memory')}
    artifact = dict(schema='awomopaw.task-artifact/v1', task=task, revision=digest(revised),
                    parent_revision=parent['revision'] if parent else None, content=revised,
                    status='candidate', split=split, run_id=run['run_id'],
                    feedback_sha256=file_digest(output / 'development-feedback.json'),
                    rationale=candidate['rationale'], validation_plan=candidate['validation_plan'],
                    qualification='Development candidate; independent evaluation required; weights unchanged')
    verify_artifact(artifact, task)
    if revised['source']:
        methods = {method.split('(')[0] for method in primitive_api.get('methods', {})}
        for node in ast.walk(ast.parse(revised['source'])):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == 'robot' and node.attr not in methods):
                raise ValueError('Task candidate uses an undocumented robot method: ' + node.attr)
    atomic_json(output / 'task-artifact.json', artifact)
    (output / 'SKILL.md').write_text(revised['skill_text'], encoding='utf-8')
    if revised['source']:
        (output / 'policy.py').write_text(revised['source'], encoding='utf-8')
    atomic_json(output / 'memory.json', revised['memory'])
    return artifact
