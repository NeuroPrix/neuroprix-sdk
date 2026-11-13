"""NeuroPrix's own trainer, with your own reward: here, a bonus for braking when the car is too fast for the corners
ahead, on top of the usual rewards.

    neuroprix-runner --server https://neuroprix.com/api --trainer "python custom_rewards.py"

How it works: neuroprix.train scores each decision with reward_for(prev_state, state, weights, steps_left, gamma).
Replacing it here (at the top of the file, so the game workers, which start by loading this file, use it too) changes
what the driver learns from every practice run. The states have everything the engine knows about the car: see the
fields of neuroprix.env.SimEnv's states (speed, distance, health, corner_over, braking, impact, contacts, ...).
"""
import neuroprix.train as T

BRAKE_BONUS = 2.0
_standard = T.reward_for


def reward_for(prev, cur, rw=None, steps_left=0, gamma=0.99):
    r = _standard(prev, cur, rw, steps_left, gamma)
    if cur.get("braking") and prev.get("corner_over", 0) > 0.05:   # braking when it was too fast for what's ahead
        r += BRAKE_BONUS
    return r


T.reward_for = reward_for

if __name__ == "__main__":
    T.main()   # reads --session session.json, trains, writes the best brain to brain_out, prints progress
