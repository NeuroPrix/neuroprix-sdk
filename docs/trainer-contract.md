# The trainer contract

A trainer is any program the runner can start. The runner prepares each session in a fresh folder and runs:

```
<your trainer command> --session /tmp/neuroprix-xxxx/session.json
```

(The folder is the trainer's working directory, and `NEUROPRIX_SESSION` holds the same path.)

## 1. Read `session.json`

```json
{
  "neuroprix_session": 1,
  "id": "a6480de05a",
  "driver": "Jango",
  "class": "n2",
  "settings": {"updates": 400, "batch": 8, "eval_every": 10, "traffic": 4, "random_tracks": true, "damage": -90.0},
  "tracks": [{"name": "My first gym", "control": [[120, 300], [164, 200], ...], "width": 90, "grip": 1, "laps": 2, "obstacles": []}],
  "start_brain": "/tmp/neuroprix-xxxx/start.json",
  "start_weights": null,
  "brain_out": "/tmp/neuroprix-xxxx/brain.json",
  "workers": 6,
  "engine_version": "2026.11.3"
}
```

| Field | Meaning |
|---|---|
| `neuroprix_session` | The format's version (1). Refuse versions you don't know. |
| `driver`, `class` | The driver's name and class (`n4`, `n3`, `n2`, `n1`): the brain must be that class's shape (see brain-format.md). |
| `settings` | What the player chose in the Garage: session length (`updates`), checks (`eval_every`), cars per practice run (`traffic`), practice on random Grand Prix circuits (`random_tracks`), `seeds` (circuit numbers), `one_way` (gyms only as drawn), and reward weights (`progress`, `fast`, `damage`, `crash`, `corner`, `overtake`, `lose_place`, …). Use what makes sense for your trainer; ignore the rest. |
| `tracks` | The gyms to practise on (Gym Editor format; `neuroprix.engine.build_track` reads them). |
| `start_brain` | Continuing a driver: its current brain (JSON). `null` for a new driver. `start_weights` is always `null` (the runner never writes PyTorch files it was sent: make your own from `start_brain`). |
| `brain_out` | Where to write your best brain. |
| `workers` | How many cores the player gave the runner. |
| `engine_version` | The engine the runner has (the server's may be newer: the runner warns). |

## 2. Write your best brain to `brain_out`

Whenever you have a brain better than the last one you wrote, write it to `brain_out` in the brain format (JSON;
the runner uploads only the JSON). Write a temporary file and rename it, so the runner never reads a
half-written file. The runner uploads every new version; when the session ends, the last one becomes the driver's
next version. If you never write one, the session ends as failed.

## 3. Optionally, print progress

Print one JSON object per line to stdout (other lines are kept as the session's log): see progress-events.md. With
`update` events the Garage shows the session's chart and how far along it is; with `eval` events it shows each check's
score and replay.

## 4. How it ends

- Exit with code 0: the session is **done** (the last brain you wrote becomes the next version).
- Exit with another code: **failed** (the last brain you wrote can still be kept: the Garage offers it).
- Stop in the Garage: the runner sends your program SIGTERM (on Windows, it ends the process tree). The last brain you
  wrote is kept if the player chose "Stop and keep".

## Tips

- `neuroprix.env.SimEnv` is the engine as a training environment: many cars on one circuit, random starts, the
  same inputs the races give, and training measurements (how far too fast for the corners ahead, hit strength).
- The default trainer is `neuroprix.train` (`python -m neuroprix.train --help`): its `collect`, `reward_for` and
  `evaluate` are good building blocks.
- Practise both ways round your gyms (`neuroprix.train.reverse_spec`), and on random circuits: drivers that only ever
  see one direction stop trying the other.
