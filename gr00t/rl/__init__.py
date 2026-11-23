"""Critic-guided RL fine-tuning of GR00T flow policies (OGPO, Best-of-N).

Kept import-light on purpose: ``gr00t.rl.envs`` is imported by simulator workers that
run in their own virtualenvs without the model stack.
"""
