#!/usr/bin/env python3
"""Paired local Pi0.5 vs GPT/Pi0.5 screening using one embodiment executor."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from cryptography.fernet import Fernet


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = "http://127.0.0.1:8098"
PROVIDER = Path("/home/ymz/.awomopaw.secret/providers/custom/gpt6.json")
TASKS = ["general_pickup", "stack_blocks_by_language",
         "classify_objects_by_language", "press_by_number"]
SYSTEM = """You are the GPT planner for the local QwenPaw RoboDojo simulation.
Use only the official task instruction, public RGB cameras, public proprioception,
and the action specification. The goal is to finish the exact official instruction.
Preserve requested object/category/color order and return-to-origin requirements.
Each decision occurs at an execution-window boundary. Prefer Pi0.5: describe
one clear visually grounded current subtask and an optimized_task_prompt which
keeps the overall goal and relevant constraints. Do not invent extra objects or
replace the requested operation with a different one. Do not repeat an unchanged
failed prompt indefinitely. Reflect on the previous public frames and decisions.
You may use exactly one absolute-joint direct action only after an attributable
provider error or repeated visually unchanged states after a prompt revision.
Ground it in public robot state, use only current action_spec fields, finite
values and stated gripper ranges. No guessed geometry, object coordinates,
unpublished joint limits, score/reward access or simulator calls. Prefer returning
to Pi0.5 after at most one correction. Abort if progress cannot be grounded.
Return ONLY a JSON object with these keys:
mode: policy | direct | abort
subtask_goal: short string
optimized_task_prompt: full concise VLA instruction (policy mode)
planning_summary: visible evidence and rationale
fallback_reason: observable provider failure (direct mode only)
actions: exactly one action object matching action_spec (direct mode only).
An ended episode must not be resumed or silently reset."""


def request(path, payload=None, timeout=240):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BRIDGE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def log(message):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), message, flush=True)


def wait_gpu():
    """Respect the other chat's active experiment and use this GPU serially."""
    announced = False
    while True:
        processes = subprocess.check_output(["ps", "-eo", "args"], text=True)
        external = any("runtime/physicalrsi-evolution/run_evolution.py" in line
                       or "simulator-entry.py" in line
                       for line in processes.splitlines())
        memory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
        ], text=True)
        if not external and int(memory.splitlines()[0]) < 2500:
            log("GPU available; local sequential evaluation may start")
            return
        if not announced:
            log("WAITING_GPU: PhysicalRSI experiment remains active")
            announced = True
        time.sleep(15)


def wait_ready(run, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request("/v1/evaluations/status?run_id=" + run["run_id"])
        if state["status"] not in {"starting", "running"}:
            raise RuntimeError("episode initialization ended: " + json.dumps(state))
        try:
            obs = request("/v1/sessions/observation", timeout=30)
            if obs.get("observation_id"):
                return obs
        except urllib.error.HTTPError as exc:
            if exc.code != 409:
                raise
        time.sleep(5)
    raise TimeoutError("initial public observation did not become ready")


def planner(obs, history, dest, boundary, direct_count, provider_error, horizon=0,
            code_policy_fallback=False):
    settings = json.loads(PROVIDER.read_text())
    api_key = settings["api_key"]
    if api_key.startswith("ENC:"):
        master_path = PROVIDER.parents[2] / ".master_key"
        master = bytes.fromhex(master_path.read_text().strip())
        api_key = Fernet(base64.urlsafe_b64encode(master)).decrypt(
            api_key[4:].encode()).decode()
    public = {k: obs[k] for k in ["task", "instruction", "observation_id",
               "action_count", "action_limit", "robot_state", "action_spec",
               "last_subtask_goal", "last_task_prompt"] if k in obs}
    public.update(history=history[-6:], direct_actions_used=direct_count,
                  direct_actions_maximum=4, provider_error=provider_error,
                  execution_window_actions=horizon or "full provider chunk")
    system = SYSTEM
    if code_policy_fallback:
        system = SYSTEM.split("You may use exactly one absolute-joint direct action")[0] + """
An additional execution backend can run a newly generated Python policy using
public RGB text segmentation, calibrated pose estimates and generic IK/movement
primitives through the same embodiment. Choose mode code_policy after repeated
VLA deviations or no visible target progress, including at least one revised
subtask/prompt. Switching does not change the benchmark instruction. Do not
generate numeric actions here; the code proposer will receive the evidence and
documented primitive API. Do not wait for the episode to end if recovery remains
possible. Use abort only when the episode or remaining budget precludes recovery.
Return ONLY JSON with mode: policy | code_policy | abort, subtask_goal,
optimized_task_prompt (policy mode), and planning_summary explaining visible
evidence and any switch reason. Never claim task success from motion alone."""
    if horizon:
        system += (
            f"\nThis run uses receding-horizon execution: at most {horizon} original "
            "provider actions per call, then discard the old tail, observe and infer again. "
            "A few windows may be needed to finish a reach or grasp; do not label "
            "ordinary intermediate movement as failure. Avoid changing a useful "
            "instruction while the same subtask is making visible progress.")
    content = [{"type": "input_text", "text": json.dumps(public, ensure_ascii=False)}]
    hashes = {}
    for camera in ["cam_head", "cam_left_wrist", "cam_right_wrist"]:
        with urllib.request.urlopen(BRIDGE + "/v1/sessions/observation/image?topic="
                                    + camera, timeout=30) as response:
            pixels = response.read()
        frame_path = dest / "frames" / f"{boundary:03d}-{camera}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(pixels)
        hashes[camera] = hashlib.sha256(pixels).hexdigest()
        content.extend([
            {"type": "input_text", "text": "Current public RGB camera: " + camera},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,"
             + base64.b64encode(pixels).decode()}])
    body = {"model": "gpt-6-astra", "input": [
        {"role": "system", "content": system},
        {"role": "user", "content": content}],
        "reasoning": {"effort": "high"}, "max_output_tokens": 4000}
    save(dest / "planning" / f"{boundary:03d}-public-input.json",
         {"system": system, "public_observation": public, "frame_hashes": hashes})
    req = urllib.request.Request(settings["base_url"].rstrip("/") + "/responses",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
        "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(req, timeout=240) as response:
        reply = json.load(response)
    save(dest / "planning" / f"{boundary:03d}-response.json", reply)
    result_text = reply.get("output_text") or "".join(
        part.get("text", "") for item in reply.get("output", [])
        for part in item.get("content", []) if part.get("type") == "output_text")
    result_text = result_text.strip()
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decision = json.loads(result_text)
    allowed_modes = {"policy", "code_policy", "abort"} if code_policy_fallback else {"policy", "direct", "abort"}
    if decision.get("mode") not in allowed_modes:
        raise ValueError("invalid planner decision mode")
    save(dest / "planning" / f"{boundary:03d}-decision.json", decision)
    return decision


def official_result(run, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request("/v1/evaluations/status?run_id=" + run["run_id"])
        if state["status"] not in {"starting", "running"}:
            return request("/v1/evaluations/result?run_id=" + run["run_id"])
        time.sleep(2)
    raise TimeoutError("evaluator did not finish after episode terminal state")


def episode(task, seed, mode, output, horizon=0):
    wait_gpu()
    active = request("/health").get("active_evaluation")
    if active and active["status"] in {"starting", "running"}:
        raise RuntimeError("another local harness session is active")
    dest = output / task / f"seed-{seed}" / mode
    run = request("/v1/sessions", {"task": task, "seed": seed, "policy_id": "pi05"})
    save(dest / "run.json", run)
    log(f"START {task} seed={seed} mode={mode} run={run['run_id']}")
    history = []
    direct_count = 0
    action_count = 0
    provider_error = None
    record = {"task": task, "seed": seed, "mode": mode, "run_id": run["run_id"],
              "status": "running", "boundaries": 0, "direct_actions": 0,
              "execution_horizon": horizon if mode != "pi05_baseline" else 0}
    try:
        obs = wait_ready(run)
        instruction = obs.get("instruction")
        if not instruction:
            raise RuntimeError("official public task instruction is unavailable")
        if isinstance(instruction, list):
            instruction = instruction[0]
        decision_budget = max(40, int(obs["action_limit"]) + 16)
        for boundary in range(decision_budget):
            save(dest / "observations" / f"{boundary:03d}.json", obs)
            if obs.get("episode_end"):
                break
            if mode == "pi05_baseline":
                decision = {"mode": "policy", "subtask_goal": instruction,
                            "optimized_task_prompt": instruction,
                            "planning_summary": "Unmodified official instruction; no LLM."}
            else:
                decision = planner(obs, history, dest, boundary, direct_count, provider_error, horizon)
            log(f"DECISION {task} {mode} boundary={boundary} actions={obs['action_count']} "
                f"mode={decision['mode']} goal={decision.get('subtask_goal', '')}")
            if decision["mode"] == "abort":
                request("/v1/sessions/abort", {"reason": decision.get("planning_summary", "planner_abort")[:500]})
                break
            if decision["mode"] == "direct":
                if direct_count >= 4:
                    request("/v1/sessions/abort", {"reason": "bounded GPT fallback budget exhausted"})
                    break
                if not provider_error and boundary < 2:
                    raise RuntimeError("direct-action requested without observed provider failure")
                chunk = request("/v1/sessions/direct-action-chunks", {
                    "observation_id": obs["observation_id"],
                    "subtask_goal": decision["subtask_goal"],
                    "fallback_reason": decision["fallback_reason"],
                    "rationale": decision.get("planning_summary", ""),
                    "actions": decision["actions"]})
                direct_count += 1
            else:
                try:
                    chunk = request("/v1/sessions/policy-chunks", {
                        "observation_id": obs["observation_id"], "provider_id": "pi05",
                        "subtask_goal": decision["subtask_goal"],
                        "optimized_task_prompt": decision["optimized_task_prompt"],
                        "planning_summary": decision.get("planning_summary", "")})
                    provider_error = None
                except urllib.error.HTTPError as exc:
                    provider_error = {"http_status": exc.code, "message": exc.read().decode()[:2000]}
                    save(dest / "provider-error.json", provider_error)
                    if mode == "pi05_baseline":
                        raise RuntimeError("baseline provider inference failed") from exc
                    history.append({"decision": decision, "provider_error": provider_error})
                    obs = request("/v1/sessions/observation")
                    continue
            save(dest / "chunks" / f"{boundary:03d}.json", chunk)
            execution = {"chunk_id": chunk["chunk_id"]}
            if horizon and mode != "pi05_baseline":
                execution.update(max_actions=horizon, discard_remaining=True)
            result = request("/v1/sessions/action-chunks/execute", execution)
            save(dest / "execution" / f"{boundary:03d}.json", result)
            action_count += result["executed"]
            history.append({"decision": decision, "executed": result.get("executed"),
                            "action_count": action_count,
                            "outcome": result.get("outcome")})
            record.update(boundaries=boundary + 1, direct_actions=direct_count,
                          action_count=action_count)
            save(dest / "receipt.json", record)
            if result.get("episode_end") or str(result.get("outcome", "")).startswith("episode_"):
                break
            obs = request("/v1/sessions/observation")
        else:
            request("/v1/sessions/abort", {"reason": "screening decision budget exhausted"})
        final = official_result(run)
        save(dest / "official-result.json", final)
        official = final["run"]["result"]
        record.update(status="completed", score=official["score"],
                      success_rate=official["success_rate"], task_succeeded=final["task_succeeded"],
                      official_result_path=final["run"]["result_path"],
                      harness_trace=final.get("harness_trace"), videos=final.get("videos", []))
    except Exception as exc:
        record.update(status="error", error=str(exc))
        try:
            current = request("/v1/evaluations/status?run_id=" + run["run_id"])
            if current["status"] in {"starting", "running"}:
                request("/v1/sessions/abort", {"reason": "screening error: " + str(exc)[:450]})
        except Exception:
            pass
    save(dest / "receipt.json", record)
    log(f"END {task} {mode}: {json.dumps(record, ensure_ascii=False)}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--execution-horizon", type=int, choices=[0, 5, 10], default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "tasks": args.tasks, "seeds": args.seeds,
                "model": "gpt-6-astra", "provider": "pi05",
                "checkpoint": "RoboDojo-sim-arx_x5-joint-0:59999",
                "robot": "arx_x5", "gpu": "local RTX 4090",
                "baseline": "Pi0.5 with unmodified public instruction through the same full-chunk executor",
                "hybrid": f"GPT-6 public-RGB planning at {'full chunk' if not args.execution_horizon else str(args.execution_horizon)+'-action'} boundaries; at most 4 direct corrective actions",
                "execution_horizon": args.execution_horizon,
                "qualification": "single-layout screening per seed, not native full-suite success rate"}
    save(args.output / "manifest.json", manifest)
    records = []
    previous = []
    if args.resume_from:
        previous = json.loads((args.resume_from / "results.json").read_text())
    for task in args.tasks:
        for seed in args.seeds:
            for mode in ["pi05_baseline", "gpt_pi05_hybrid"]:
                completed = next((r for r in previous if r["task"] == task and
                                  r["seed"] == seed and r["mode"] == mode and
                                  r["status"] == "completed" and
                                  (mode == "pi05_baseline" or
                                   r.get("execution_horizon", 0) == args.execution_horizon)), None)
                if completed:
                    source_root = Path(completed.get("reused_receipt_root", args.resume_from))
                    completed["reused_receipt_root"] = str(source_root)
                    completed["previous_results_source"] = str(args.resume_from / "results.json")
                    source = source_root / task / f"seed-{seed}" / mode / "execution"
                    completed["action_count"] = sum(
                        json.loads(p.read_text())["executed"] for p in source.glob("*.json"))
                    records.append(completed)
                    save(args.output / "results.json", records)
                    log(f"REUSE {task} {mode} run={completed['run_id']}")
                    continue
                record = episode(task, seed, mode, args.output, args.execution_horizon)
                records.append(record)
                save(args.output / "results.json", records)
                time.sleep(8)
    lines = ["# Local paired RoboDojo screening", "",
             f"Same checkpoint and seed; baseline executes full chunks, hybrid uses horizon={args.execution_horizon or 'full'}. Scores are official.", "",
             "| Task | Seed | Pi0.5 score | GPT + Pi0.5 score | Delta |", "|---|---:|---:|---:|---:|"]
    for task in args.tasks:
        for seed in args.seeds:
            pair = {r["mode"]: r for r in records if r["task"] == task and r["seed"] == seed}
            a = pair["pi05_baseline"].get("score")
            b = pair["gpt_pi05_hybrid"].get("score")
            lines.append(f"| {task} | {seed} | {a} | {b} | {b-a if a is not None and b is not None else 'not comparable'} |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
