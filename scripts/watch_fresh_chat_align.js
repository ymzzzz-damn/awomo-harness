// Read-only compact status for the running fresh-Chat evaluation.
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const root = process.argv[2];
  const summaryPath = path.join(root, 'summary.json');
  const summary = fs.existsSync(summaryPath) ? JSON.parse(fs.readFileSync(summaryPath)) : {};
  const scenes = fs.readdirSync(root).filter(n => n.startsWith('scene-')).sort();
  const scene = scenes.at(-1);
  const events = fs.readFileSync(path.join(root, scene, 'stream.sse'), 'utf8')
    .split('\n').filter(l => l.startsWith('data:')).map(l => {
      try { return JSON.parse(l.slice(5)); } catch { return {}; }
    });
  const calls = events.filter(e => e.type === 'plugin_call' && e.status === 'completed')
    .flatMap(e => e.content?.map(c => c.data) ?? []);
  const start = calls.find(c => c.name === 'robodojo_sim__start_session');
  let selected;
  try { selected = JSON.parse(start?.arguments ?? '{}').policy_id; } catch {}
  const health = await (await fetch('http://127.0.0.1:8098/health')).json();
  const active = health.active_evaluation;
  let actions;
  let sources;
  if (active) {
    const trace = path.join(active.artifact_dir, 'awomopaw_harness_trace.jsonl');
    if (fs.existsSync(trace)) {
      const rows = fs.readFileSync(trace, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
      const executions = rows.filter(e => e.event === 'action_chunk_execution');
      actions = executions.at(-1)?.action_count ?? 0;
      sources = executions.reduce((s, e) => (s[e.source] = (s[e.source] ?? 0) + e.executed, s), {});
    }
  }
  console.log(JSON.stringify({finished: summary.finished ?? 0, successes: summary.task_successes ?? 0,
    skill_successes: summary.automatic_skill_successes ?? 0, latest_scene: scene,
    selected_provider: selected, tool_turns: calls.length, last_tool: calls.at(-1)?.name,
    active_run: active?.run_id, actions, sources}));
})().catch(e => { console.error(e.message); process.exitCode = 1; });
