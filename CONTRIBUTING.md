# Contributing

Thanks for helping. Bug reports, fixes, docs and new example trainers are all welcome.

## Reporting a bug

Open an issue with:

- what you ran and what it printed (remove your runner token if it appears anywhere),
- `neuroprix-runner --version`, your operating system and `python --version`.

Security problems: please report them privately (see [SECURITY.md](SECURITY.md)), not in an issue.

Please don't post the contents of `~/.neuroprix/runner.json`: it holds your runner's token. If a token leaks, remove
that computer in the Garage; the token stops working at once.

## Running the tests

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"        # on Windows: .venv\Scripts\pip install -e ".[test]"
.venv/bin/pytest
```

The tests need no NeuroPrix server: `tests/conftest.py` has a fake one that speaks the runner protocol. They train
tiny sessions for every class with the real trainer, so they take a few minutes.

## How changes reach the SDK

The engine, the trainer and the runner in this repository are the same code the NeuroPrix server and game use. They're
copied here for each release, so the SDK always matches the races. A pull request to `neuroprix/engine.py`,
`neuroprix/env.py`, `neuroprix/brain.py`, `neuroprix/train.py` or `neuroprix/runner.py` is welcome, but it will be
applied upstream and arrive here in the next release rather than being merged directly. Docs, examples and tests can
be merged directly.

Physics changes (`neuroprix/engine.py`) can't be accepted on their own: the races must use the same engine for
everyone, so they're decided with the game.

## Style

Plain, simple language in docs and messages. Code: keep the dependencies to NumPy and PyTorch, and the runner to
Python's standard library.
