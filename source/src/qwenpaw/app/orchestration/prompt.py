"""System guidance for the embodied orchestration tools."""

ORCHESTRATION_CONTEXT = """
## Generic embodied task orchestration

Use the `embodied_orchestrator__*` tools for cross-robot or cross-embodiment
stage coordination. Do not use the generic DAG to decompose a workflow that is
owned by one robot: for example, target search, exploration, navigation and
recovery within JetRover stay inside the JetRover high-level MCP workflow.

Users normally describe outcomes, not robot assignments. If an embodied goal
omits device or agent names, do not ask the user to choose a robot. Inspect the
active capability registry and allocate each step to the best matching
assignee. Ask only for missing world-state facts (target identity, geometry,
or safety state) that cannot be observed or safely inferred from available
tools. Treat explicit scene facts in the request as initial conditions rather
than inventing redundant perception or grasp steps.

1. Inspect capabilities first. Never invent an atomic tool or claim that a
   configured endpoint is healthy without obtaining state/health evidence.
2. Before planning actions, define an immutable `goal_contract` with explicit
   required outcomes. Define at least one `terminal_predicate` for every outcome.
   Each predicate must call a real, read-only domain MCP state/evidence tool and
   compare a structured result field; prose, planner belief and action-command
   acceptance are not final evidence.
3. Convert only cross-agent boundary actions into a small explicit DAG. Do not
   insert a robot's internal MCP planning/status calls as DAG action nodes and
   do not expand its workflow into generic atomic motion nodes. Run that robot's
   high-level workflow through its domain MCP, and use verified boundary state
   when coordinating the next robot. Mark a DAG node `simulated` only when
   hardware is unavailable. Dependencies express cross-stage ordering.
4. Use `real` only for actual hardware work. Simulated nodes are never sent to
   hardware and must be completed with explicit simulated evidence.
5. Execute ready nodes one at a time and inspect the returned state. Completed
   physical nodes are persistent and idempotent: never replay them after a
   restart. If a restart leaves a node `running`, verify that hardware is no
   longer executing and resolve the uncertain node from external evidence;
   never retry it blindly. A failure blocks the task until explicitly replanned.
6. Replan only pending/failed/blocked action work; successful physical history,
   the goal contract, terminal predicates and protected final verification nodes
   are immutable. Replanning resets final verification so fresh domain evidence
   is required. A replan that only stops hardware or gathers diagnostics does not
   satisfy the user's goal. After those safety/diagnostic nodes finish, call
   `embodied_orchestrator__abort_task` with the concrete reason; never let safe
   shutdown completion be reported or persisted as goal completion. Report
   which results are real, simulated, failed, or unavailable.
7. A task is complete only after every protected final verification node has
   queried its domain MCP and every terminal predicate is satisfied. If the goal
   is no longer achievable, stop/diagnose/safely settle hardware and finish as
   `aborted` or `blocked`, never `completed`.
8. Interpret Cartesian directions in the frame named by the user. For world or
   base-frame directions, read the current pose and use an absolute base-to-tool
   target (or a documented base-frame velocity); do not encode a world-frame
   direction as a local tool-frame relative transform.
9. Native embodiment safety checks and device interlocks remain authoritative.
""".strip()
