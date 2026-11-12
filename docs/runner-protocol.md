# The runner protocol

How a runner talks to the NeuroPrix server: JSON over HTTP(S), always started by the runner (it works behind home
routers; the server never connects to it). You only need this to write your own runner; the SDK's does all of it.

## Linking (once)

1. `POST /runner/pair` `{"name": "My PC", "info": {...}}` → `{"pair_id", "secret", "code", "expires_in"}`. Show the code.
2. The player types the code in the Garage (Train tab → Add a computer).
3. `POST /runner/pair/poll` `{"pair_id", "secret"}` every few seconds → `{"status": "waiting"}` until
   `{"status": "linked", "runner_id", "token", "name"}`. The token is shown once: keep it private. Every call below
   sends it as the header `X-Runner-Token`.

## Working

- `POST /runner/heartbeat` `{"info": {"cpus", "workers", "platform", "version", "engine", "trainer"}}` every few
  seconds → `{"job": null | {...}, "engine_version"}`. A job is a session to train: `id`, `name`, `tier` (class),
  `flags` (the settings as trainer options, e.g. `["--updates", "400", "--random-tracks"]`), `tracks`,
  `resume_json` (the starting brain as JSON, or null; `resume_pth` is always null, and the SDK's runner ignores it). While training, send `{"job_id": ...}` instead: the
  answer says whether to stop. A runner that checks in idle while the server thinks it's training a session ends
  that session (it was lost, e.g. the runner restarted).
- `POST /runner/jobs/{id}/report` `{"lines": [...]}`: what the trainer printed (progress events and log lines), a
  batch at a time → `{"stop": bool, "keep": bool}` (Stop pressed in the Garage).
- `PUT /runner/jobs/{id}/brain` `{"brain_json": "..."}`: the best brain so far (JSON only: the server keeps just the weights and the driver's name). Refused
  (422) unless it's readable and the driver's class.
- `POST /runner/jobs/{id}/finish` `{"status": "done" | "failed" | "stopped", "error": "..."}`.

A session whose runner goes quiet for 2 minutes ends as failed, keeping its best brain. `401` means the runner was
removed from the account: link it again.
