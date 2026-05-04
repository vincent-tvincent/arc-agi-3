import random

import arc_agi
from arcengine import GameState

arc = arc_agi.Arcade()
print("Available environments:")
for env_info in arc.get_environments():
    print("-", env_info.game_id, env_info.title)

env = arc.make("ls20", render_mode="terminal-fast", save_recording=True)
if env is None:
    raise SystemExit("Could not create ls20 environment")

print("Action space:", env.action_space)

for step in range(20):
    action = random.choice(env.action_space)
    obs = env.step(action)

    print("step", step, "action", action, "state", obs.state if obs else None)

    if obs and obs.state == GameState.WIN:
        print("Won!")
        break

print("Scorecard:", arc.get_scorecard())
