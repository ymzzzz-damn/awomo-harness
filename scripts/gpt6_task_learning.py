#!/usr/bin/env python3
"""Success skills and success/failure video feedback for the next same-task policy.

This workflow uses the connected AwomoPaw model provider directly. It does not
bundle an external controller, replay an episode, or consume native action steps.
"""
import argparse
import asyncio
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from gpt6_failure_learning import (MODEL, SCHEMA as FAILURE_SCHEMA, digest, save,
    empty_memory, validate_memory, validate_analysis, working_memory_view,
    learning_eligibility, failure_evidence, normalize_visual_observations)

SCHEMA = deepcopy(FAILURE_SCHEMA)
SCHEMA['properties']['outcome_summary'] = SCHEMA['properties'].pop('failure_summary')
SCHEMA['required'].remove('failure_summary')
SCHEMA['required'].append('outcome_summary')
for key in ('success_mechanisms', 'policy_improvements'):
    SCHEMA['properties'][key] = {'type': 'array', 'items': {'type': 'string'}}
    SCHEMA['required'].append(key)
SCHEMA['properties']['skill_text'] = {'type': 'string'}
SCHEMA['required'].append('skill_text')

INSTRUCTIONS = '''Rethink a completed DEVELOPMENT episode using its verified official
outcome, timestamped samples of all three public camera videos, executed policy
history, public receipts and logs. Source artifacts are reference data. Never use
hidden simulator geometry, memorize scene coordinates or replace the official goal.
Separate visible observations from causal hypotheses. Sparse video cannot prove
why the evaluator accepted/rejected a trajectory; state uncertainty explicitly.
Cite supplied frame IDs. Use at most 8 observations, 4 hypotheses, 5 lessons.
Each lesson needs an observable trigger, applicable correction/check and limits.
On SUCCESS: explain plausible successful mechanisms, retain useful prior failure
checks, and distill a task skill as Markdown beginning with a heading (no YAML).
Describe reusable subtasks, current-RGB checks, action budget and completion checks;
avoid replaying literal targets or presenting the last worker as the whole solution.
On FAILURE: diagnose causes and propose next-policy corrections; skill_text must
be empty and success_mechanisms must be []. Do not replace an existing success skill.
For BOTH outcomes update bounded task memory and propose evidence-based policy
improvements, for the next scene of the SAME task. Success is not proof that each
worker or every motion was necessary. Keep outcome_summary <=1500 characters,
each observation/hypothesis/lesson/mechanism/improvement <=1500, task memory <=10000,
skill_text <=8000. These are development references, not independent qualification.
Return only the requested JSON. Do not train weights or restart the task.'''


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def empty_task_memory(experiment, task, artifact=None):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,127}', task):
        raise ValueError('Invalid task name')
    memory = dict(empty_memory(experiment), schema='awomopaw.task-experience/v2',
                  task=task, task_artifact=artifact)
    if artifact:
        loaded = load_task_artifact(artifact, task)
        memory['task_memory'] = loaded['memory']['task_memory']
    return memory


def artifact_revision(task, parent_revision, skill_text, memory, provenance):
    return content_digest(dict(task=task, parent_revision=parent_revision,
        skill_text=skill_text, memory=memory, provenance=provenance))


def load_task_artifact(reference, task):
    path = Path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Task artifact manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'awomopaw.task-artifact/v2' or manifest.get('task') != task:
        raise ValueError('Task artifact belongs to another task or schema')
    if set(manifest['files']) != {'SKILL.md', 'memory.json'}:
        raise ValueError('Task artifact requires SKILL.md and memory.json')
    for name, expected in manifest['files'].items():
        if digest(path.parent/name) != expected:
            raise ValueError('Task artifact content changed: ' + name)
    skill_text = (path.parent/'SKILL.md').read_text()
    memory = json.loads((path.parent/'memory.json').read_text())
    if memory['task'] != task or not manifest['provenance'].get('successful_run_ids'):
        raise ValueError('A reusable skill requires successful same-task provenance')
    revision = artifact_revision(task, manifest['parent_revision'], skill_text, memory, manifest['provenance'])
    if revision != manifest['revision'] or revision != reference['revision']:
        raise ValueError('Task artifact revision mismatch')
    return dict(manifest=manifest, skill_text=skill_text, memory=memory,
                directory=path.parent, reference=reference)


def validate_task_memory(memory, experiment, next_layout, task=None):
    validate_memory(memory, experiment, next_layout)
    if memory.get('schema') != 'awomopaw.task-experience/v2' or (task and memory.get('task') != task):
        raise ValueError('Same-task experience schema mismatch')
    if memory.get('task_artifact'):
        load_task_artifact(memory['task_artifact'], memory['task'])


def validate_rethinking(value, frame_ids, success):
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Outcome rethinking schema mismatch')
    legacy = {k: value[k] for k in FAILURE_SCHEMA['required'] if k != 'failure_summary'}
    legacy['failure_summary'] = value['outcome_summary']
    validate_analysis(legacy, frame_ids)
    for key in ('success_mechanisms', 'policy_improvements'):
        values = value[key]
        if not isinstance(values, list) or len(values) > 5 or any(
                not isinstance(v, str) or not v.strip() or len(v) > 1500 for v in values):
            raise ValueError('Invalid ' + key)
    if not isinstance(value['skill_text'], str) or len(value['skill_text']) > 8000:
        raise ValueError('Invalid/budget-exceeding skill text')
    if success:
        if not value['success_mechanisms'] or not value['skill_text'].lstrip().startswith('#'):
            raise ValueError('Success needs mechanisms and a reusable Markdown skill')
    elif value['skill_text'] or value['success_mechanisms']:
        raise ValueError('Failure feedback cannot invent a successful skill')


def verified_outcome(case_dir, task):
    row = json.loads((case_dir/'receipt.json').read_text())
    official = json.loads((case_dir/'official-result.json').read_text())
    run = official['run']
    if (not row.get('valid_episode') or type(row.get('success')) is not bool or
            run['status'] != 'completed' or run['run_id'] != row['run_id'] or run['task'] != task or
            type(official.get('task_succeeded')) is not bool or official['task_succeeded'] != row['success']):
        raise ValueError('A matching completed official task outcome is required')
    native = run['result']
    if native.get('eval_time') != 1 or len(native.get('details', {})) != 1 or native['score'] != row['score']:
        raise ValueError('Expected one verified scored development episode')
    detail = next(iter(native['details'].values()))
    if detail.get('layout_id') != row['layout_id'] or detail.get('success') != row['success']:
        raise ValueError('Official scene/outcome mismatch')
    return row


def distill_after(before, row, analysis, evidence, output):
    """One outcome advances memory once; failures retain the successful skill."""
    output = Path(output)
    parent = load_task_artifact(before['task_artifact'], before['task']) if before.get('task_artifact') else None
    entry = dict(source_layout_id=row['layout_id'], source_run_id=row['run_id'],
        official_success=row['success'], analysis_sha256=digest(output/'analysis.json'),
        trace_sha256=evidence['trace_sha256'], evidence_sha256=digest(output/'evidence.json'))
    lessons = before.get('active_lessons', []) + [dict(id=f"E{row['layout_id']:02d}-L{i+1:02d}",
        source_layout_id=row['layout_id'], outcome='success' if row['success'] else 'failure', text=text)
        for i, text in enumerate(analysis['lessons'])]
    after = dict(before, version=before['version']+1, entries=before['entries']+[entry],
        task_memory=analysis['updated_task_memory'], active_lessons=lessons[-8:])
    if row['success'] or parent:
        prior = parent['manifest'] if parent else None
        success_ids = prior['provenance']['successful_run_ids'] if prior else []
        mechanisms = list(parent['memory'].get('success_mechanisms', [])) if parent else []
        if row['success']:
            success_ids = success_ids + [row['run_id']]
            mechanisms.append(dict(source_run_id=row['run_id'], mechanisms=analysis['success_mechanisms']))
            skill = ('---\nname: learned-' + before['task'].replace('_', '-') + '\n'
                     'description: Learned development strategy for ' + before['task'] + '.\n---\n\n' +
                     analysis['skill_text'].strip() + '\n')
        else:
            skill = parent['skill_text']
        memory = dict(schema='awomopaw.task-memory/v2', task=before['task'],
            task_memory=after['task_memory'], active_lessons=after['active_lessons'],
            success_mechanisms=mechanisms[-8:], policy_improvements=analysis['policy_improvements'],
            source_run_id=row['run_id'], source_outcome='success' if row['success'] else 'failure')
        provenance = dict(entry, successful_run_ids=success_ids,
            scope='development_reference_for_next_same_task_policy', independent_qualification=False)
        parent_revision = prior['revision'] if prior else None
        revision = artifact_revision(before['task'], parent_revision, skill, memory, provenance)
        directory = output/'task-artifact'
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'SKILL.md').write_text(skill)
        save(directory/'memory.json', memory)
        manifest = dict(schema='awomopaw.task-artifact/v2', task=before['task'], revision=revision,
            parent_revision=parent_revision, provenance=provenance,
            status='success_distillation' if row['success'] else 'failure_feedback_update',
            files={name: digest(directory/name) for name in ['SKILL.md', 'memory.json']})
        save(directory/'task-artifact.json', manifest)
        after['task_artifact'] = dict(path=str((directory/'task-artifact.json').resolve()),
            sha256=digest(directory/'task-artifact.json'), revision=revision)
    validate_task_memory(after, before['experiment'], row['layout_id']+1, before['task'])
    working_memory_view(after)
    save(output/'experience-after.json', after)
    return after


async def analyze(evidence, before, config_file, output):
    from qwenpaw.providers import ProviderManager
    slot = json.loads(Path(config_file).read_text())['active_model']
    if slot['model'] != MODEL:
        raise ValueError('Rethinking must use the same configured GPT6 model')
    provider = ProviderManager.get_instance().get_provider(slot['provider_id'])
    if provider is None:
        raise RuntimeError('Pinned model provider unavailable')
    parent = load_task_artifact(before['task_artifact'], before['task']) if before.get('task_artifact') else None
    payload = dict(task=before['task'], prior_task_memory=before['task_memory'],
        prior_success_skill=parent['skill_text'] if parent else '',
        prior_success_mechanisms=parent['memory'].get('success_mechanisms', []) if parent else [],
        evidence={k: v for k, v in evidence.items() if k != 'frames'})
    content = [dict(type='input_text', text=json.dumps(payload, ensure_ascii=False))]
    for f in evidence['frames']:
        content += [dict(type='input_text', text=f"Frame {f['id']} at {f['timestamp_seconds']:.3f}s"),
            dict(type='input_image', detail='high', image_url='data:image/jpeg;base64,' +
                 base64.b64encode(Path(f['path']).read_bytes()).decode())]
    save(output/'reflection-input.json', dict(instructions=INSTRUCTIONS, **payload,
        frame_manifest=evidence['frames'], model=MODEL, frame_count=len(evidence['frames'])))
    client = provider._client(timeout=180).with_options(max_retries=0)
    errors = []
    try:
        for attempt in range(3):
            response = await client.responses.create(model=MODEL,
                instructions=INSTRUCTIONS + ('\nCorrect validation error: ' + errors[-1] if errors else ''),
                input=[dict(role='user', content=content)],
                text={'format': {'type': 'json_schema', 'name': 'task_video_rethinking', 'strict': True, 'schema': SCHEMA}})
            save(output/f'response-{attempt}.json', response.model_dump(mode='json'))
            with (output/'model-audit.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(requested_model=MODEL, returned_model=response.model,
                    response_id=response.id, frame_count=len(evidence['frames']), attempt=attempt))+'\n')
            if not (response.model == MODEL or str(response.model).startswith(MODEL+'-')):
                raise RuntimeError('Rethinking model mismatch')
            try:
                value, omitted = normalize_visual_observations(json.loads(response.output_text))
                if omitted:save(output/f'trace-observations-{attempt}.json', omitted)
                validate_rethinking(value, {f['id'] for f in evidence['frames']}, evidence['terminal_result']['success'])
                return value
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
        raise RuntimeError('Rethinking output validation failed: ' + '; '.join(errors))
    finally:
        await client.close()


async def learn(case_dir, snapshot, config_file, output, split='development', model_analysis=analyze):
    if split != 'development':
        raise ValueError('Only development outcomes may feed learning')
    case_dir, snapshot, output = Path(case_dir), Path(snapshot), Path(output)
    before = json.loads(snapshot.read_text())
    row = verified_outcome(case_dir, before['task'])
    validate_task_memory(before, before['experiment'], row['layout_id'], before['task'])
    qualification = learning_eligibility(case_dir, allow_success=True)
    if not qualification['eligible']:
        raise ValueError('Task learning refused: ' + qualification['reason'])
    initial = json.loads((case_dir/'initial-observation.json').read_text())
    if (initial.get('task') != before['task'] or initial.get('run_id') != row['run_id'] or
            not isinstance(initial.get('instruction'), str) or not initial['instruction'].strip()):
        raise ValueError('Initial public observation does not match the completed task')
    output.mkdir(parents=True, exist_ok=False)
    save(output/'eligibility.json', qualification)
    evidence = failure_evidence(case_dir, allow_success=True, output=output)
    analysis = await model_analysis(evidence, before, config_file, output)
    validate_rethinking(analysis, {f['id'] for f in evidence['frames']}, row['success'])
    save(output/'analysis.json', analysis)
    after = distill_after(before, row, analysis, evidence, output)
    save(output/'receipt.json', dict(status='completed', source_layout_id=row['layout_id'],
        source_run_id=row['run_id'], outcome='success' if row['success'] else 'failure',
        input_version=before['version'], output_version=after['version'], before_sha256=digest(snapshot),
        after_sha256=digest(output/'experience-after.json'), frame_count=len(evidence['frames']),
        model=MODEL, task_artifact=after.get('task_artifact'), no_episode_replay=True))
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'validation', 'final'], default='development')
    args = parser.parse_args()
    asyncio.run(learn(args.case_dir, args.snapshot, args.agent_config, args.output, args.split))


if __name__ == '__main__':
    main()
