#!/usr/bin/env python3
"""VLA failure -> generated Python policy, retaining the native episode/executor."""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import compare_local_tasks as control


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/awomopaw-code-policy"
sys.path.insert(0, str(ROOT / "PhysicalRSI"))
sys.path.insert(0, str(ROOT / "runtime/physicalrsi-evolution"))
from run_evolution import LLM
from PhysicalRSI_baselines.robodojo.proposal import CodeProposer
from PhysicalRSI_baselines.robodojo.rethinking import VideoRethinker, failed_episodes


def public(obs):
    return {k: obs[k] for k in ("task", "instruction", "observation_id", "run_id", "action_count",
                              "action_limit", "robot_state", "initial_robot_state", "action_spec") if k in obs}


def frames(output, label):
    paths = []
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        import urllib.request
        with urllib.request.urlopen(control.BRIDGE + "/v1/sessions/observation/image?topic=" + camera, timeout=30) as r:
            data = r.read()
        path = output / "frames" / f"{label}-{camera}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths.append(path)
    return paths


def propose(output, obs, history, source, memory, attempts):
    api = control.request("/v1/sessions/code-policy-api")
    api.update(task=obs["task"], instruction=obs["instruction"],
               remaining_action_budget=obs["action_limit"] - obs["action_count"])
    model = LLM()
    model.output = output
    model.images = frames(output, "takeover")
    model.images += [Path(a["final_public_frame"]) for a in attempts[-2:]
                    if a.get("final_public_frame") and Path(a["final_public_frame"]).is_file()]
    feedback = [{"task": obs["task"], "split": "development",
                 "current_public_observation": public(obs),
                 "vla_decisions_and_execution": history,
                 "previous_generated_attempts": attempts,
                 "note": "The benchmark goal is the unmodified observation instruction. This is public execution evidence and completed development outcomes, not hidden simulator geometry."}]
    artifact = CodeProposer(model).propose(task=obs["task"], source=source, memory=memory,
                                         feedback=feedback, skill_api=api, output=output)
    spec = importlib.util.spec_from_file_location("code_contract", RUNTIME / "contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_source(artifact["source"])
    (output / "generated-policy.py").write_text(artifact["source"])
    control.save(output / "proposal.json", artifact)
    return artifact


def run_worker(output, obs, artifact, reason):
    job = control.request("/v1/sessions/code-policies", {
        "observation_id": obs["observation_id"], "source": artifact["source"],
        "memory": artifact["memory"], "fallback_reason": reason[:1000],
    })
    control.save(output / "code-job.json", job)
    control.log("CODE_TAKEOVER run=" + job["run_id"] + " policy=" + job["policy_id"])
    last = None
    deadline = time.monotonic() + 950
    while time.monotonic() < deadline:
        status = control.request("/v1/sessions/code-policies/status?policy_id=" + job["policy_id"])
        if status["status"] != last:
            control.log("CODE_STATUS " + status["status"])
            last = status["status"]
        if status["status"] != "running":
            control.save(output / "code-status.json", status)
            source_dir = Path(job["artifact_dir"])
            shutil.copytree(source_dir, output / "code-runtime", dirs_exist_ok=True)
            return status
        time.sleep(3)
    raise TimeoutError("code-policy supervisor did not finish")


def close_run(run, reason):
    try:
        obs = control.request("/v1/sessions/observation")
        if not obs.get("episode_end"):
            control.request("/v1/sessions/abort", {"reason": reason[:500]})
    except Exception:
        pass
    return control.official_result(run)


def rethink_failed_run(final, obs, source, memory, output):
    run = final['run']
    return [VideoRethinker(LLM()).reflect(
        task=obs['task'], instruction=obs['instruction'], run_id=run['run_id'],
        episode=episode, result=run['result'], videos=final.get('videos', []),
        source=source, memory=memory, output=output / 'rethinking' / f'episode-{int(episode):03d}',
        status=run['status']) for episode in failed_episodes(run['result'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="align_blocks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-policy-from", type=Path,
                        help="Replay an unchanged generated policy after an infrastructure-only failure")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for _ in range(30):
        try:
            if control.request("/health", timeout=5).get("status") == "ready":
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("simulation bridge did not become ready")
    control.save(args.output / "experiment.json", {
        "task": args.task, "seed": args.seed, "maximum_code_revisions": args.rounds,
        "goal_source": "native current_observation.instruction, unchanged",
        "new_policy_source": "configured LLM via unchanged official CodeProposer",
        "prebuilt_task_policy_loaded": False, "manual_task_hint_or_policy_edits": False,
        "control_owner": "same RoboDojo embodiment HTTP executor",
        "infrastructure_policy_resume": str(args.resume_policy_from) if args.resume_policy_from else None,
        "qualification": "Experimental single-layout development; after terminal failures each revision starts a recorded fresh episode.",
    })
    source = "def policy(robot, memory):\n    return None\n"
    memory, attempts, records = {}, [], []
    history, reason = [], ""
    for revision in range(1, args.rounds + 1):
        control.wait_gpu()
        output = args.output / f"revision-{revision:02d}"
        output.mkdir()
        run = control.request("/v1/sessions", {"task": args.task, "seed": args.seed, "policy_id": "pi05"})
        control.save(output / "run.json", run)
        control.log(f"START revision={revision} run={run['run_id']}")
        terminal_vla_failure = False
        reflectable = False
        try:
            obs = control.wait_ready(run)
            if not obs.get("instruction") or not obs.get("initial_robot_state"):
                raise ValueError("native benchmark goal/origin is unavailable")
            control.save(output / "initial-observation.json", public(obs))
            if revision == 1:
                history = []
                for boundary in range(20):
                    control.save(output / "observations" / f"{boundary:03d}.json", public(obs))
                    decision = control.planner(obs, history, output, boundary, 0, None, 10,
                                               code_policy_fallback=True)
                    control.log(f"VLA_DECISION boundary={boundary} mode={decision['mode']} goal={decision.get('subtask_goal')}")
                    if decision["mode"] == "code_policy":
                        prompts = {h["decision"].get("optimized_task_prompt") for h in history}
                        if len(history) < 2 or len(prompts) < 2:
                            raise ValueError("code takeover requires repeated VLA execution and a prompt revision")
                        reason = decision["planning_summary"]
                        control.save(output / "switch.json", {"reason": reason, "evidence": history,
                                     "takeover_action_count": obs["action_count"], "same_episode": True})
                        break
                    if decision["mode"] == "abort":
                        raise RuntimeError("planner aborted: " + decision.get("planning_summary", ""))
                    chunk = control.request("/v1/sessions/policy-chunks", {
                        "observation_id": obs["observation_id"], "provider_id": "pi05",
                        "subtask_goal": decision["subtask_goal"],
                        "optimized_task_prompt": decision["optimized_task_prompt"],
                        "planning_summary": decision.get("planning_summary", ""),
                    })
                    result = control.request("/v1/sessions/action-chunks/execute", {
                        "chunk_id": chunk["chunk_id"], "max_actions": 10, "discard_remaining": True})
                    control.save(output / "vla-execution" / f"{boundary:03d}.json", result)
                    history.append({"decision": decision, "executed": result["executed"], "outcome": result["outcome"]})
                    if result.get("episode_end"):
                        terminal_vla_failure = True
                        raise RuntimeError("VLA ended the episode before code takeover")
                    obs = control.request("/v1/sessions/observation")
                else:
                    raise RuntimeError("no code switch within the VLA action budget")
            generation = output / "generation"
            generation.mkdir()
            if revision == 1 and args.resume_policy_from:
                artifact = json.loads((args.resume_policy_from / "proposal.json").read_text())
                assert (args.resume_policy_from / "generated-policy.py").read_text() == artifact["source"]
                shutil.copytree(args.resume_policy_from, generation, dirs_exist_ok=True)
                control.save(generation / "infrastructure-replay.json", {
                    "source": str(args.resume_policy_from), "reason": "Generic dependency repair; prior worker executed zero policy actions",
                    "source_sha256": hashlib.sha256(artifact["source"].encode()).hexdigest(),
                })
            else:
                artifact = propose(generation, obs, history, source, memory, attempts)
            source, memory = artifact["source"], artifact["memory"]
            # Re-read the public observation after generation. The worker is
            # pinned to this run and original goal, never to an old episode.
            obs = control.request("/v1/sessions/observation")
            status = run_worker(output, obs, artifact, reason or "Previous generated code failed in a recorded development episode")
            if isinstance(status.get("memory"), dict):
                memory = status["memory"]
            final = close_run(run, "generated code ended without completing the benchmark goal")
            control.save(output / "official-result.json", final)
            record = {"revision": revision, "run_id": run["run_id"], "status": "completed",
                      "worker_status": status["status"], "score": final["run"]["result"]["score"],
                      "task_succeeded": final["task_succeeded"], "code_actions": status.get("executed_actions"),
                      "videos": final.get("videos", []), "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            feedback = {"revision": revision, "source": source, "worker_status": status,
                        "terminal_development_result": final["run"]["result"]}
            events = output / "code-runtime/execution/events.jsonl"
            if events.is_file():
                all_events = [json.loads(s) for s in events.read_text().splitlines()]
                feedback["public_primitives_and_observations"] = [e for e in all_events if e["event"] != "code_action_executed"]
                feedback["final_action_executions"] = [e for e in all_events if e["event"] == "code_action_executed"][-5:]
                final_frames = sorted((events.parent).glob("head-*.png"))
                feedback["final_public_frame"] = str(final_frames[-1]) if final_frames else None
            attempts.append(feedback)
            reflectable = status['status'] in {'returned', 'episode_terminal'}
        except Exception as exc:
            record = {"revision": revision, "run_id": run["run_id"], "status": "error", "error": str(exc)}
            final = close_run(run, str(exc))
            control.save(output / "official-result.json", final)
            attempts.append({"revision": revision, "error": str(exc), "terminal_development_result": final["run"].get("result")})
            if terminal_vla_failure:
                record.update(status='completed', score=final['run']['result']['score'],
                              task_succeeded=final['task_succeeded'], videos=final.get('videos', []))
                reflectable = True
        reflection_error = None
        if reflectable and not final['task_succeeded']:
            try:
                reflections = rethink_failed_run(final, obs, source, memory, output)
                attempts[-1]['failure_rethinking'] = reflections
                record['rethinking_feedback'] = [str(output / 'rethinking' /
                    f"episode-{int(r['episode']):03d}" / 'harness-feedback.json') for r in reflections]
            except Exception as exc:
                reflection_error = str(exc)
                record['rethinking_error'] = reflection_error
                control.save(output / 'rethinking-error.json', {'error': reflection_error})
        control.save(args.output / 'development-history.json', attempts)
        records.append(record)
        control.save(args.output / "results.json", records)
        control.log("END " + json.dumps(record, ensure_ascii=False))
        if record.get("task_succeeded"):
            if not terminal_vla_failure:
                (args.output / "selected-policy.py").write_text(source)
            break
        if reflection_error or not reflectable:
            # Infrastructure failures and incomplete diagnosis are not strategy
            # evidence. Preserve the official result and stop before new proposals.
            break
        reason = "Previous generated policy failed; revise from recorded development observations and commands."
    control.save(args.output / "development-history.json", attempts)


if __name__ == "__main__":
    main()
