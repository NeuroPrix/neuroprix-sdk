# The NeuroPrix SDK guide

Everything you need to train your NeuroPrix drivers on your own computer, or on a machine you rent, in plain steps.

- [Quick start](#quick-start): from nothing to your first training session in about 10 minutes
- [How it works](#how-it-works): drivers, classes, sessions, versions and checks
- [The runner](#the-runner): every option, and leaving it running
- [Configuration](#configuration): environment variables and files
- [Your computer](#your-computer): CPU, GPU, Apple Silicon, Windows, Linux, rented machines and Docker
- [Training without the server](#training-without-the-server): experiments on your own
- [Write your own trainer](#write-your-own-trainer)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Quick start

You need **Python 3.10 or newer** and about **1 GB of disk space** (most of it is PyTorch). Check your Python with
`python3 --version` (on Windows: `py --version`).

### 1. Install

Make a folder for it and install the SDK into its own virtual environment, so it doesn't mix with other Python
software on your computer:

**macOS and Linux**

```bash
python3 -m venv ~/neuroprix
~/neuroprix/bin/pip install neuroprix
```

**Windows** (PowerShell)

```powershell
py -m venv $HOME\neuroprix
& $HOME\neuroprix\Scripts\pip install neuroprix
```

On Linux, you can save about 2 GB of downloads by installing the CPU-only build of PyTorch first (the SDK doesn't use
a graphics card; see [Your computer](#your-computer)):

```bash
~/neuroprix/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/neuroprix/bin/pip install neuroprix
```

Check it worked:

```bash
~/neuroprix/bin/neuroprix-runner --version
```

It prints something like `neuroprix-runner 0.2.0 (engine 2026.11.3)`.

### 2. Start the runner and link it to your account

The runner is the program that trains on your computer. Start it with your NeuroPrix server's address: the site's
address followed by `/api` (on the site's **Train** page, **Add a computer** shows it):

```bash
~/neuroprix/bin/neuroprix-runner --server https://neuroprix.com/api
```

The first time, it shows a code:

```
========================================================
  Link this runner to your NeuroPrix account:

  1. Open https://neuroprix.com and go to the Train page
  2. Pick a driver; under "Trains on", choose "Add a computer"
  3. Type this code:   K7QF-2MXP

  (valid for 10 minutes)
========================================================
```

In your browser, sign in and open the **Train** page (in the bar at the top), then pick a driver. Under **Trains
on**, click **Add a computer**, type the code and click **Link a runner**. The runner prints `linked as "your-computer". Waiting for training sessions.`

You only do this once per computer: the runner remembers the link.

### 3. Train a driver

Leave the runner running. On the **Train** page, with the driver picked:

1. Choose what to practise on: your gyms, or **Surprise tracks** (a new random circuit every practice run, the best
   preparation for races on circuits the driver hasn't seen).
2. Pick a session length: **Quick** (60 updates), **Standard** (150) or **Long** (400).
3. Under **Trains on**, choose your computer.
4. Click **Start training**.

The runner picks the session up within about 10 seconds and prints `training <driver> (<class>), session <id>`.
You watch it on the Train page as usual: the progress chart, a replay of each check, and **Stop and keep the best
version** if you want to end early. When the session ends, the best version becomes the driver's new version, ready
to race.

That's it. Leave the runner running and it will train every session you send to it, one at a time.

**How much training does a driver need?** A brand-new driver starts out crashing at the first corner. On a simple
gym it gets going within one or two Standard sessions; to get round Grand Prix circuits it usually needs a few hundred
updates (N4: about 200–400; the other classes more, N1 about 800). Don't worry if the first sessions' checks barely move: queue a
**Long** session, or a few in a row. Each session continues from the driver's best version so far.

---

## How it works

### Drivers and classes

A **driver** is a small neural network (its **brain**) that looks at what the car senses and decides what to do,
several times a second. You're the coach: you choose how it practises; the training makes it better.

Every driver belongs to a **class**. The class decides what the driver can sense and do, and so the shape of its brain:

| Class | What the driver controls | Brain shape |
|---|---|---|
| N4 | Steering only (the throttle is automatic) | 18 inputs → 64 → 3 actions |
| N3 | Steering, throttle and brake | 22 → 64 → 9 |
| N2 | N3 plus a battery: boost and slipstream | 24 → 64 → 12 |
| N1 | N2 on terrain: hills, crests and banked corners, which it senses ahead | 31 → 64 → 12 |

The outputs above are one score per move (the `combined` action layout). A brain can instead pick its steering, its
pedal and (N2, N1) its boost separately: `two_head` (N3 6, N2 / N1 7 outputs) or `three_head` (N2 / N1, 8). **New N1
brains are `three_head`**; a driver that keeps training keeps its layout unless you move it on the Train page. See
[action layouts](brain-format.md#action-layouts).

Those are today's brains. The next generation of each class swaps the radar on the 2 nearest cars ahead for one that
sees all round the car, behind too: 20 more inputs (N4 38, N3 42, N2 44, N1 51; `--senses sectors` trains it). A class
moves to it on a set date ([observations-and-actions.md](observations-and-actions.md#the-sector-radar-the-next-generation-of-brain)).

N4 learns fastest; N1 has the most to learn. You choose the class with the **Class** switch at the top of the site
when you create the driver, and it doesn't change.
The details of what each class senses: [observations-and-actions.md](observations-and-actions.md).

### Sessions

A **training session** is one go at improving a driver. You start it on the Train page; the runner trains it. A session:

1. **Practises.** Many cars, all driven by this driver, race around the practice circuits. Each round of practice is
   an **update**: a batch of practice runs (8 by default), after which the brain is adjusted a little towards what
   went well. A Standard session is 150 updates, so 1,200 practice runs.
2. **Checks.** Every 10 updates (you can change this), the driver drives the check circuits on its own, at full
   speed, without any randomness. That's the score you see on the Train page, with a replay of the drive.
3. **Keeps the best.** Whenever a check beats every earlier one, that brain is kept and sent to the server. So a
   session never makes a driver worse: if the later updates go badly, you keep the best version from earlier.

Practice uses rewards: points for covering ground, for speed and clean overtakes; penalties for damage, crashes,
corners taken too fast and lost places. On the Train page, **Driving style** and **All settings…** let you change them;
the runner passes your settings to the trainer.

### Versions

Each session that ends with a brain gives the driver a new **version**. The next session starts from the driver's
current version and carries on improving it (or, if you tick the option to start fresh, from a blank brain). A
driver's **History** tab on the Train page lists every session and version.

### What goes back to the server

Only three things travel from your computer to the server, all over HTTPS and started by your computer:

- **Progress**: the lines the trainer prints (updates, checks and their replays), for the Train page's live view.
- **Brains**: each new best brain (a small JSON file, about 50 KB). The server checks that it's a brain of the
  driver's class and refuses anything else.
- **How the session ended**: done, stopped from the Train page, or failed (with the error).

From the server, the runner takes only data: the session's settings (each checked against the list of known
settings; anything unknown is refused), the gyms, and the driver's current brain. **It never runs code the server
sends.** Races always run on the server, with the same engine for everyone, so nothing on your computer can change a
race result. A better-trained brain just drives better.

---

## The runner

```
neuroprix-runner [--server URL] [--name NAME] [--workers N] [--trainer COMMAND] [--relink] [--version]
```

| Option | What it does | Default |
|---|---|---|
| `--server URL` | The NeuroPrix server: the site's address followed by `/api` (`https://neuroprix.com/api`). Remembered after the first time, so later you can leave it out. | The remembered one |
| `--name NAME` | What this computer is called in the Train page's **Trains on** list. Only used when linking. | This computer's name |
| `--workers N` | How many practice games to run at the same time. More is faster but uses more of your computer. | CPU cores minus 2 (at most 12) |
| `--trainer COMMAND` | Train with your own trainer instead of NeuroPrix's, e.g. `--trainer "python my_trainer.py"`. See [Write your own trainer](#write-your-own-trainer). | NeuroPrix's PPO trainer |
| `--relink` | Forget the current link and show a new code (for example, to link this computer to a different account). | |
| `--allow-http` | Allow a plain `http://` server that isn't this computer. Unsafe: the runner's token travels unencrypted. Only for a server on your own network. | off |
| `--version` | Print the runner's version and its engine version, and exit. | |
| `--help` | List the options. | |

`python -m neuroprix.runner` is the same as `neuroprix-runner`.

### What the runner prints

```
22:41:07 NeuroPrix Runner 0.2.0: "my-pc" on https://neuroprix.com/api, 6 games in parallel, trainer: NeuroPrix PPO. Waiting for training sessions (Ctrl+C to quit).
22:43:15 training Dex Jango (N2), session a6480de05a
22:44:02   update 25
22:44:51   update 50
...
22:49:30 session a6480de05a done
22:49:30 waiting for training sessions
```

### Stopping

- **Stop a session**: use **Stop and keep the best version** on the Train page. The runner stops training, the driver
  keeps the best version found so far, and the runner waits for the next session.
- **Stop the runner**: press **Ctrl+C** (or `docker stop`, or stop the service). If it was training, that session
  ends straight away and the driver keeps its best version so far. Nothing is lost except the training that hadn't
  yet produced a better version.

### If the connection drops or the computer sleeps

- **Short outages** (Wi-Fi drops, the server restarts): training carries on. The runner keeps the progress and sends
  it when the connection is back. You'll see `can't reach the server (...); still training, will send progress when
  it's back`.
- **Long outages** (more than about 2 minutes, or the computer goes to sleep): the server stops waiting and ends the
  session as failed, **keeping the best version sent so far**. Start a new session on the Train page to carry on from
  there: it continues from the driver's current version. Sessions don't resume halfway through; a new session picks
  up from the last kept version.
- **The runner restarted** while training: when it comes back, the server ends that session the same way (best
  version kept), and the runner is ready for a new one.

To stop a laptop from sleeping during long sessions: on macOS, run the runner with `caffeinate -i` in front
(`caffeinate -i ~/neuroprix/bin/neuroprix-runner`); on Windows, set **Sleep** to **Never** while plugged in; on Linux,
`systemd-inhibit ~/neuroprix/bin/neuroprix-runner`.

### Several computers, one account

Link as many as you like (each with its own code). Each trains one session at a time. On the Train page, pick which one
trains each session under **Trains on**. A session sent to a computer that's offline or busy waits for it.

### Removing a computer

On the Train page, click **Remove** next to the computer under **Trains on**. Its link stops working immediately; if
it's running, it says `this runner was removed from your account; link it again` and shows a new code.

---

## Configuration

The runner needs no configuration file: the options above are enough. These environment variables are available
for scripts, services and containers:

| Variable | Same as | Example |
|---|---|---|
| `NEUROPRIX_SERVER` | `--server` | `https://neuroprix.com/api` |
| `NEUROPRIX_TRAINER` | `--trainer` | `python /work/my_trainer.py` |
| `NEUROPRIX_RUNNER_CONFIG` | Where the link is kept | `/data/runner.json` |
| `NEUROPRIX_TRAIN_WORKERS` | `neuroprix-train --workers` (when training without the runner) | `4` |

### The link file

The link to your account is kept in `~/.neuroprix/runner.json` (on Windows,
`C:\Users\<you>\.neuroprix\runner.json`). It holds the server address, this runner's name and its **token**: a key
that lets this computer train your drivers (only that: it can't sign in as you or change anything else). The file
is readable only by you. **Don't share it or commit it anywhere.** If it leaks, remove the computer on the Train page:
the token stops working at once. Deleting the file (or `--relink`) makes the runner ask for a new code.

### Where sessions run

Each session trains in a fresh temporary folder, deleted when the session ends. Nothing is kept on your computer
after a session: the brains are on the server.

---

## Your computer

### What makes training fast

Training time goes into **simulating the races**, not into the neural network (the brains are tiny). So what matters
is the number of **CPU cores**: each worker plays its own practice games on one core. The runner uses all cores but
two by default, leaving some for you. Use `--workers` to change that:

- Leave your computer usable while it trains: `--workers 2` or `--workers 4`.
- A machine that does nothing else: `--workers` equal to its number of cores.

As a rough guide: in our tests a Standard N4 session (150 updates) took about 4 minutes with only 2 workers on a
busy Apple Silicon Mac; more workers make it faster. A Long session (400) takes about three times as long, and N3, N2
and N1 take longer per update than N4 (N1 the longest).

### Graphics cards (GPU)

**Not needed and not used.** The brain is 64 neurons wide; moving it to a graphics card would be slower than keeping
it on the CPU. A computer with more cores beats one with a better graphics card. This also means the CPU-only build
of PyTorch is all you need.

### Apple Silicon Macs (M1 to M4)

Work well: `pip install neuroprix` installs PyTorch for Apple Silicon. Use the Python from
[python.org](https://www.python.org/downloads/) or Homebrew (`brew install python@3.12`). The performance cores do
most of the work, so `--workers` set to the number of performance cores is a good choice (for example 8 on an M3 Pro).

### Intel Macs

PyTorch stopped making new versions for Intel Macs after 2.2. Use **Python 3.12 or older**; pip then installs
PyTorch 2.2, which works.

### Windows

Works with Python from [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH**) or the
Microsoft Store. Run the commands in PowerShell, using `Scripts\` instead of `bin/` (see the Quick start). Ctrl+C
stops the runner as on other systems.

### Linux

Any recent distribution. Install the CPU-only PyTorch first (see the Quick start) to save a large download.

### A rented machine (cloud)

Any machine with several CPU cores works; a compute-optimised one with 8–32 cores is a good choice. GPU machines cost
more and don't train faster. Rent it, install the SDK (or use Docker, below), start the runner with
`--server https://neuroprix.com/api`, and type the code it shows on the Train page. To keep it running after you log
out, use `tmux` or `screen`, or Docker with `--restart unless-stopped`.

### Docker

The repository has a `Dockerfile` with the CPU-only PyTorch:

```bash
git clone <this repository> neuroprix-sdk && cd neuroprix-sdk
docker build -t neuroprix-runner .
docker run -it -v neuroprix:/home/runner/.neuroprix neuroprix-runner --server https://neuroprix.com/api
```

The volume (`-v neuroprix:/home/runner/.neuroprix`) keeps the link, so the container doesn't ask for a new code each time it
starts. After linking, run it in the background:

```bash
docker run -d --restart unless-stopped --name neuroprix -v neuroprix:/home/runner/.neuroprix neuroprix-runner
```

`docker logs -f neuroprix` shows what it's doing; `docker stop neuroprix` ends it cleanly (a session it was training
ends, keeping its best version).

---

## Training without the server

`neuroprix-train` is NeuroPrix's trainer on its own, for experiments: try settings, compare reward ideas, watch the
numbers. It trains on your computer and writes the brain to a file. (To give a driver a new version, train it
through the runner: that's how brains reach the site.)

```bash
neuroprix-train --tier n3 --random-tracks --updates 60 --out experiment.pth
```

This writes the best brain to `experiment.json` (and `experiment.pth`, the same weights for PyTorch), plus
`experiment.latest.*`, the newest weights whether or not they were the best. Useful options (`neuroprix-train --help` lists all):

| Option | What it does | Default |
|---|---|---|
| `--tier n4\|n3\|n2\|n1` | The class | `n3` |
| `--updates N` | Session length | 150 |
| `--batch N` | Practice runs per update | 8 |
| `--random-tracks` | A new random Grand Prix circuit every practice run | off |
| `--seeds 3,8,21` | Practise on these numbered circuits | circuit 1 |
| `--track gym.json` | Practise on a gym exported from the Gym Editor (repeatable) | |
| `--traffic N` | Cars per practice run, all driven by this driver (1 = alone) | 4 |
| `--eval-every N` | Check every N updates | 10 |
| `--resume brain.json` | Start from an existing brain of the same class (a `.json`, or a `.pth` with PyTorch 2.6 or newer) | a new brain |
| `--layout combined\|two_head\|three_head` | The brain's [action layout](brain-format.md#action-layouts). A resumed brain can move forward (combined → two_head → three_head) and keeps what it learned | a resumed brain's own; a new N1 brain `three_head`; others `combined` |
| `--workers N` | Games in parallel | cores minus 2 (at most 8) |
| `--progress-json` | Print progress as JSON lines (as the runner sees it) | text |
| reward weights | `--progress`, `--fast`, `--damage`, `--crash`, `--corner`, `--overtake`, `--lose-place`, `--wall`, … | NeuroPrix's defaults |
| learning | `--lr`, `--entropy`, `--gamma`, `--lam`, `--clip`, `--epochs`, `--minibatch` | NeuroPrix's defaults |

The engine also works on its own in Python, for your own experiments:

```python
from neuroprix.env import SimEnv

env = SimEnv()
env.setup_world({"seed": 202}, cars=4, tier="n3")   # Grand Prix circuit no. 202, 4 cars, class N3
states = env.reset_world()                          # what each car senses: states[i]["obs"]
states = env.step_world([0, 0, 0, 0])               # one decision for each car (4 physics frames)
print(states[0]["distance"], states[0]["dead"], states[0]["finished"])
```

---

## Write your own trainer

The runner can run any trainer, written in any language, that follows a small contract:

1. It's started as `<your command> --session session.json`. The file says which driver and class, the settings, the
   gyms, the starting brain and where to write the result.
2. It writes its best brain to the path in `brain_out`, whenever it has a better one.
3. Optionally, it prints progress events (one JSON object per line) for the Train page's chart and replays.

```bash
neuroprix-runner --trainer "python my_trainer.py"
```

Start from one of the examples in [`examples/`](../examples):

- [`custom_rewards.py`](../examples/custom_rewards.py): NeuroPrix's trainer with one reward of your own added.
- [`minimal_trainer.py`](../examples/minimal_trainer.py): a complete trainer in about 100 lines of NumPy, showing the
  whole contract.

Test it without the server by writing a `session.json` yourself (the format is in
[trainer-contract.md](trainer-contract.md)) and running `python my_trainer.py --session session.json`.

The details: [trainer-contract.md](trainer-contract.md), [brain-format.md](brain-format.md),
[progress-events.md](progress-events.md), [observations-and-actions.md](observations-and-actions.md). To write a
runner of your own: [runner-protocol.md](runner-protocol.md).

---

## Updating

```bash
~/neuroprix/bin/pip install -U neuroprix
```

Then restart the runner (it keeps its link). Update when the runner prints:

```
WARNING: this runner's engine is 2026.11.2, the server's is 2026.12.1: update the runner (pip install -U neuroprix) ...
```

That means the race physics changed: until you update, your drivers would practise on different physics than the
races use. The Train page flags such runners too. What changed in each version: [CHANGELOG.md](../CHANGELOG.md). About
engine versions: [engine.md](engine.md).

---

## Troubleshooting

**`pip install neuroprix` fails with "No matching distribution found for torch"**
Your Python is too new or too old for PyTorch, or you're on an Intel Mac with Python 3.13+. Use Python 3.10–3.12
(Intel Macs) or 3.10–3.14 (others): `python3.12 -m venv ~/neuroprix`, then install again.

**`requires a different Python: 3.9.6 not in '>=3.10'`**
Your Python is too old (macOS's built-in `/usr/bin/python3` is 3.9). Install a newer one from python.org or Homebrew
and create the virtual environment with it.

**`command not found: neuroprix-runner`**
Use the full path (`~/neuroprix/bin/neuroprix-runner`, or `$HOME\neuroprix\Scripts\neuroprix-runner` on Windows), or
activate the environment first (`source ~/neuroprix/bin/activate`; on Windows `& $HOME\neuroprix\Scripts\Activate.ps1`).

**`... isn't encrypted (http://): use https://`**
The runner only sends its token over an encrypted connection, except to a server on this computer (`localhost`).
Use the `https://` address.

**`the server redirected to https://...: use that address as --server`**
The runner doesn't follow redirects (so its token only ever goes to the address you gave). Use the address it names.

**`--server must be a web address starting with https://`**
Include `https://`: `--server https://neuroprix.com/api`, not `--server neuroprix.com/api`.

**`can't get a code from https://... ; retrying in 10 s`**
The runner can't reach the server. Check the address (it ends in `/api`: `https://neuroprix.com/api`), your internet connection, and that
a firewall or proxy isn't blocking Python. It keeps trying, so it links as soon as the server is reachable.

**The code expired**
Codes last 10 minutes. The runner shows a new one by itself.

**The Train page says "Type the code your runner shows" or the code isn't accepted**
Type it exactly as shown (the dash is optional; upper or lower case both work), within 10 minutes, while signed in
to the account the drivers are on.

**`this runner was removed from your account; link it again`**
The computer was removed on the Train page (or its link was reset). Type the new code it shows.

**The session stays "waiting for its runner"**
Check that the runner is running and says `Waiting for training sessions`, and that the session was sent to *this*
computer (under **Trains on**). A runner trains one session at a time: a second one waits for the first to finish.
The Train page says when a computer is offline or busy.

**`the runner refused this session: unknown setting ...: this runner may be out of date`**
The server uses a training setting this version doesn't know. Update: `pip install -U neuroprix`.

**`the server refused the brain: ...`**
The trainer wrote a brain of the wrong shape for the driver's class. With NeuroPrix's trainer, this shouldn't happen
(please report it); with your own, check the shapes in [brain-format.md](brain-format.md).

**`session ... failed: ...`**
The trainer stopped with an error; the message is its last lines. With your own trainer, run it by hand with the
same `session.json` to see the whole error. With NeuroPrix's trainer, please report it with the message.

**Training is slow, or my computer is sluggish**
Lower `--workers` to keep the computer responsive; raise it (up to the number of cores) for speed. Close other heavy
programs. On a laptop, plug it in: on battery, the system slows the CPU down.

**The session failed with "the runner was restarted while training this session"**
The runner (or computer) restarted mid-session. The best version so far was kept; start a new session to carry on.

**Behind a company proxy**
Python reads the standard `HTTPS_PROXY` environment variable: `HTTPS_PROXY=http://proxy:8080 neuroprix-runner`.

**Starting over**
Delete `~/.neuroprix/runner.json` (or run with `--relink`) and remove the computer on the Train page.

---

## FAQ

**Does the runner cost anything?**
No. It uses your computer's power, or a machine you rent and pay for directly.

**Is it safe to run?**
The runner only ever runs the trainer you chose (NeuroPrix's, which comes with this package, or yours). The server
sends data, never code: every setting it sends is checked against a fixed list, brains travel as JSON numbers only
(never as PyTorch files, which can carry code), and text from the server is cleaned before it's printed. The runner
opens no ports: it connects out to the server, like a browser, only over HTTPS (except to a server on this computer),
and never follows a redirect elsewhere. In Docker it runs as an ordinary user, not root. It's open source, so you can
read every line. To report a security problem, see [SECURITY.md](../SECURITY.md).

**Can someone use my computer through it?**
No. The server can only ask it to train *your* drivers with the known settings. The token in the link file only lets
a runner train your drivers; it can't sign in to your account.

**Can I train someone else's drivers, or let someone train mine?**
A runner trains the drivers of the account it's linked to. To help a friend, they link your runner to their account
(with `--relink` and their code), and it then trains only their drivers.

**Does a stronger computer win races?**
It trains faster, so you can train more in the same time. Races run on the server, with the same engine and rules for
everyone, and the brain's size is fixed per class. Coaching (what to practise, which rewards, how long) matters as
much as compute.

**Can I modify the trainer or the engine?**
The trainer: yes, that's what the SDK is for (see [Write your own trainer](#write-your-own-trainer)). The engine
copy in the SDK only changes your practice: races always use the server's engine, so changing your copy only makes
your practice less like the real races.

**Can I train while I use my computer?**
Yes. Use `--workers 2` (or so) to leave most of the computer free.

**What happens to a session if I close my laptop?**
It ends, keeping the best version sent so far. Start a new session to carry on from it.

**Do I need to keep the browser open?**
No. Once the session is sent to your computer, you can close the browser. Open the Train page any time to see progress.

**Where are my brains stored?**
On the NeuroPrix server, with your drivers. Nothing stays on your computer after a session.

**Can I see the code?**
Yes: this repository. The license is Apache 2.0 (see [LICENSE](../LICENSE)).

**I found a bug.**
Please open an issue in this repository with what you ran, what it printed (remove your token if it appears), your
operating system and `neuroprix-runner --version`.
