"""Simulation code-policy contract; no benchmark-specific manipulation sequence."""

import ast


METHODS = {
    "reobserve()": "Return instruction, current state and episode initial_state. State contains left/right_arm_joint_state, left/right_ee_joint_state and left/right_ee_pose=[x,y,z,w,x,y,z]. World meters and wxyz quaternions.",
    "detect(text, count=1, plane_z=None)": "Segment a text-named object in the current head RGB with SAM3. Return up to count estimates sorted by world x. Each has position_m, yaw_rad, long_axis_xy, extent_xy_m and confidence. Positions are calibrated RGB ray/plane estimates, not simulator object poses. Returns [] if no match.",
    "approach(arm, position_m, quaternion_wxyz, steps=12)": "CuRobo collision-aware trajectory to a world end-effector pose. arm is left or right. May fail. steps is action count, not duration.",
    "servo(arm, position_m, quaternion_wxyz=None, steps=12)": "IK and interpolated joints for a nearby world end-effector pose. Default orientation is the observed EE quaternion. May fail.",
    "move_joints(arm, joints, steps=12)": "Interpolate a finite absolute six-joint target, reobserving after each action.",
    "gripper(arm, command, steps=3)": "Normalized gripper: 1 open, 0 closed. Hold other joints.",
    "home(arm='left', steps=15)": "Return to joints observed at the beginning of this benchmark episode, not at the takeover point.",
    "hold(steps=3)": "Hold current joint/gripper targets. Every target still goes through the embodiment executor.",
    "contact_to_wrist(position_m, quaternion_wxyz)": "Convert a fingertip contact xyz to an EE wrist xyz using the robot's 0.135m local-x fingertip offset.",
    "quaternion_from_euler(xyz_radians)": "Euler xyz radians to quaternion wxyz.",
}
API = {
    "schema": "awomopaw.sim-code-policy/v1",
    "policy_signature": "def policy(robot, memory)",
    "methods": METHODS,
    "available_python": "No imports. Use arithmetic, conditionals, bounded loops, builtin abs/min/max/sum/len/range/enumerate/zip/sorted/list/dict/tuple/float/int/str/bool/round/all/any and Exception/ValueError. Dictionary get/items/keys/values/copy and list append are available. Only documented robot methods are allowed.",
    "calibration": {"robot": "arx_x5", "frame": "world", "table_height_m": 0.765,
                    "default_rgb_object_plane_m": 0.785, "fingertip_offset_m": 0.135,
                    "note": "Static robot/camera calibration and an approximate perception plane; no layout coordinates."},
    "boundary": "RGB and public robot state only. No file/network/simulator access from generated code. The worker translates generic primitives into one validated action at a time through the existing embodiment. The evaluator alone decides success.",
}


def validate_source(source):
    if not isinstance(source, str) or not 1 <= len(source) <= 60000:
        raise ValueError("source must contain 1 to 60000 characters")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(n.name == "policy" and [a.arg for a in n.args.args] == ["robot", "memory"] for n in functions):
        raise ValueError("source must define policy(robot, memory)")
    forbidden = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "dir", "help", "type", "object"}
    robot_methods = {m.split("(")[0] for m in METHODS}
    containers = {"get", "append", "items", "keys", "values", "copy"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Global)):
            raise ValueError("code policy may not import modules or define privileged scopes")
        if isinstance(n, ast.Name) and (n.id in forbidden or n.id.startswith("__")):
            raise ValueError("forbidden code-policy capability: " + n.id)
        if isinstance(n, ast.Attribute):
            if n.attr.startswith("_"):
                raise ValueError("private attribute access is forbidden")
            if isinstance(n.value, ast.Name) and n.value.id == "robot":
                if n.attr not in robot_methods:
                    raise ValueError("undocumented robot method: " + n.attr)
            elif n.attr not in containers:
                raise ValueError("only documented robot and container attributes are allowed")
    return tree
