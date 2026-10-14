"""Lifecycle-only stub. GPT6 authors every action through the harness."""
class Model:
    def __init__(self, *args, **kwargs): pass
    def reset(self, *args, **kwargs): pass
    def update_obs(self, *args, **kwargs):
        raise RuntimeError('No VLA inference in GPT6-only evaluation')
    def get_action(self, *args, **kwargs):
        raise RuntimeError('GPT6 must submit actions through harness')
    def close(self, *args, **kwargs): pass
