# Security

## Reporting a problem

Please don't open a public issue for a security problem. Use **Report a vulnerability** on this repository's
**Security** tab (GitHub's private reporting), with what you found and how to reproduce it. We'll reply there.

## What the runner protects

- **Your runner's token** (in `~/.neuroprix/runner.json`) lets a computer train your drivers, nothing more: it can't
  sign in as you or change your account. The file is created readable only by you. The token is only sent over
  HTTPS (plain HTTP only to a server on the same computer, or with `--allow-http`), and the runner never follows a
  redirect, so the token only goes to the address you linked. If it leaks, remove that computer on the Train page: the
  token stops working at once.
- **Your computer.** The runner connects out to the server and opens no ports. The server sends data only: every
  training setting is checked against a fixed list, ids are checked before they're used, brains travel as JSON
  numbers (PyTorch files, which can carry code, are never loaded from the server), and the server's text is cleaned
  of terminal control codes before it's printed. The only program the runner starts is the trainer you chose.
- **Fair races.** Races run on the server. A runner can only send brains of the right shape for the driver's class,
  and the server keeps only their weights.

## What's yours to look after

- A trainer you pass with `--trainer` runs with your user's permissions: only use code you trust.
- Keep the SDK up to date (`pip install -U neuroprix`) to get fixes.
