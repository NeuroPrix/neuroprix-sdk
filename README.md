# NeuroPrix SDK

Train your [NeuroPrix](https://neuroprix.com) drivers on your own computer, or on a machine you rent.

NeuroPrix is a championship for AI racing drivers: you're the coach, and a small neural network is the driver. Most
players train in the browser. With this SDK, a computer of yours does the training: it can run all day with all its
cores, and you can close the browser. If you want, you can also change *how* your drivers learn: your own rewards,
your own practice drills, a different learning algorithm. The official races always run on the NeuroPrix server with
the same rules for everyone; what you control is how your driver's brain is trained.

**→ Full guide: [docs/guide.md](docs/guide.md)** (install, linking, training, every option, hardware, troubleshooting, FAQ)

## Quick start

You need Python 3.10 or newer.

```bash
python3 -m venv ~/neuroprix                     # a separate environment for it
~/neuroprix/bin/pip install neuroprix           # the SDK and PyTorch (about 1 GB)
~/neuroprix/bin/neuroprix-runner --server https://neuroprix.com/api
```

(On Windows: `py -m venv $HOME\neuroprix`, then `$HOME\neuroprix\Scripts\pip` and `...\Scripts\neuroprix-runner`.)

1. The runner shows a code. On the site, open the **Train** page and pick a driver. Under **Trains on**, click
   **Add a computer**, type the code and click **Link a runner**. (Once per computer: it remembers the link.)
2. Under **Trains on**, choose your computer, then click **Start training**.
3. The runner trains the session; you watch it on the Train page as usual (progress, the replay of each check, Stop and
   keep). The best version becomes your driver's new version.

Leave the runner running: it trains every session you send it, one at a time. Ctrl+C stops it.

With Docker (good for a rented machine):

```bash
docker build -t neuroprix-runner .
docker run -it -v neuroprix:/home/runner/.neuroprix neuroprix-runner --server https://neuroprix.com/api
```

## What's in it

| Part | What it is |
|---|---|
| `neuroprix-runner` (`neuroprix.runner`) | Links a computer to your account, takes the sessions you start on the Train page, runs a trainer and sends back its progress and brains |
| `neuroprix-train` (`neuroprix.train`) | NeuroPrix's own trainer (PPO): what the runner uses by default, and a baseline to run on its own, copy or change |
| `neuroprix.engine`, `neuroprix.env` | The race engine: the same physics, circuits and senses the official races use (versioned) |
| `neuroprix.brain` | Brains: the network for each class, saving and loading |

## Classes

| Class | Driver | Brain |
|---|---|---|
| N4 | Steering only (automatic throttle) | 18 → 64 → 3 |
| N3 | Steering, throttle and brake | 22 → 64 → 9 |
| N2 | N3 plus a battery: boost and slipstream | 24 → 64 → 12 |
| N1 | N2 on terrain: hills, crests and banked corners, which it senses ahead | 31 → 64 → 12 |

(Until November 2026 NeuroPrix was called NeuroRacer, and N4/N3/N2 were F3/F2/F1; the old codes are still accepted.)

## Train your own way

The runner runs a **trainer**: any program that follows a small contract ([docs/trainer-contract.md](docs/trainer-contract.md)):

1. It's started with `--session session.json`. The file says which driver and class, the session's settings, the
   gyms to practise on, the starting brain (when continuing a driver) and where to write the result.
2. It trains, and whenever it has a better brain, writes it to `brain_out` (the brain format:
   [docs/brain-format.md](docs/brain-format.md)). The runner uploads each new version; the last one becomes the
   driver's next version.
3. Optionally, it prints progress events, one JSON object per line ([docs/progress-events.md](docs/progress-events.md)),
   for the Train page's live chart and the replay of each check.

```bash
neuroprix-runner --trainer "python my_trainer.py"
```

Start from the default trainer (`neuroprix-train --help`) or from the examples:

- [examples/custom_rewards.py](examples/custom_rewards.py): the default trainer with your own reward for each step.
- [examples/minimal_trainer.py](examples/minimal_trainer.py): a complete trainer in about 100 lines (random search
  with NumPy, no PyTorch), to show the whole contract.

The engine works on its own too, for experiments:

```python
from neuroprix.env import SimEnv
env = SimEnv()
env.setup_world({"seed": 202}, cars=4, tier="n3")     # a Grand Prix circuit, 4 cars, class N3
states = env.reset_world()                              # what each car senses: states[i]["obs"]
states = env.step_world([0, 0, 0, 0])                   # one decision each (4 physics frames)
```

What a driver senses and can do in each class: [docs/observations-and-actions.md](docs/observations-and-actions.md).

## The rules that don't change

- **The brain's shape is fixed per class.** Train however you like; the server refuses a brain of the wrong shape or
  class.
- **The engine is versioned** (`neuroprix.ENGINE_VERSION`). When the physics changes, update the SDK
  (`pip install -U neuroprix`): the runner warns you, and the Train page flags a runner whose engine is out of date,
  since it would be training on different physics than the races use. Big changes come between seasons.
- **Races run on the server.** Nothing you change in the SDK changes a race; a better-trained brain just drives better.
- **The server sends data, never code.** The runner checks every setting it's sent against a fixed list and only
  ever runs the trainer you chose.

## More

- [docs/guide.md](docs/guide.md): the full guide.
- [docs/runner-protocol.md](docs/runner-protocol.md): how a runner talks to the server (to write your own runner).
- [docs/engine.md](docs/engine.md): the engine's versions, and keeping up with physics changes.
- [CHANGELOG.md](CHANGELOG.md): what changed in each version.
- [CONTRIBUTING.md](CONTRIBUTING.md): running the tests, reporting bugs.
- [SECURITY.md](SECURITY.md): what the runner protects, and reporting a security problem privately.

## License

Apache License 2.0 (see [LICENSE](LICENSE)). The NeuroPrix name isn't covered by the license (see [NOTICE](NOTICE)).
