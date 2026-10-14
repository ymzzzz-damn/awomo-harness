"""Full native CuRobo trajectory/IK backend driven by official proprioception."""
from pathlib import Path
import numpy as np
from .curobo_planner import CuroboPlanner

ASSETS = Path(__file__).parent / 'assets' / 'x5'
class Robot:
    def __init__(self):
        self.bases = {a: [x, -.45, .765, .707, 0., 0., .707] for a,x in [('left',-.3),('right',.3)]}
        names = [f'joint{i}' for i in range(1,7)]
        self.planners = {a: CuroboPlanner(p,names,names,'cuda:0',dt=.004,yml_path=str(ASSETS/'curobo.yml')) for a,p in self.bases.items()}
    def update(self, state):
        self.state = state
    def solve_ik(self, arm, pose):
        return self.planners[arm].solve_ik_to_joint(self.state[f'{arm}_arm_joint_state'],pose,real_robot_pose=self.bases[arm])
    def path(self, arm, start, pose, constraint):
        result = self.planners[arm].plan_path(start,pose,self.bases[arm],constraint_pose=constraint)
        if result['status'] != 'Success':
            raise RuntimeError(f'H07 trajectory failed: {result}')
        return np.asarray(result['position'],dtype=float)
