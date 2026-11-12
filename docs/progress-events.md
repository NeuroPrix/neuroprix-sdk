# Progress events

A trainer may print these, one JSON object per line. All are optional; the more you send, the more the Garage shows.

| Event | Fields | Shown as |
|---|---|---|
| `start` | `name`, `tracks` (a label), `updates` (planned), `batch`, `traffic`, `tier`, `eval_spec` | The session's header |
| `stage` | `message` | A line under the session title ("Learning to drive…") |
| `update` | `update` (1, 2, …), `avg_return`, `avg_distance`, `car_contacts`, `elapsed` (seconds) | The progress bar and chart |
| `eval` | `update`, `progress` (a score 0..1), `saved` (true if this check's brain was written to `brain_out`), `results` (one per circuit: `track`, `progress`, `finished`, `health`, `reason`, `time`, `field_progress`, `car_contacts`), `path` (the replay: `[x, y, heading, health, braking]` per decision), `preview` (`spec`: the circuit; `others`: the other cars' paths) | A check: the score, and the replay in the Train tab |
| `holdout` | `progress`, `results` | "On circuits it has never seen" |
| `note` | `message` | The session's log |
| `done` | | |

Example:

```json
{"event": "update", "update": 12, "avg_return": 341.5, "avg_distance": 2650.0, "car_contacts": 0.8, "elapsed": 9.4}
```

The default trainer prints all of them (`python -m neuroprix.train --progress-json …`).
