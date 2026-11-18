"""RaceBrain: the small policy network that drives a NeuroPrix car.

Classes (the brain's shape says which it is), all sensing only what a real car could measure, and the heading
(docs/observations-and-actions.md; backend/sim.py Car.obs):
  N4  46 inputs -> 3 actions   steering only, automatic throttle
  N3  47 inputs -> 9 actions   steering x pedals (accelerate / coast / brake); + the steering wheel
  N2  49 inputs -> 12 actions  N3 + battery and slipstream; pedals accelerate / coast / brake / BOOST
  N1  57 inputs -> 12 actions  N2 + its tilt and 6 sight lines over the hills (or 8 with three heads, its default)
Internally a brain's "generation" is its kind: 16 = N4, 17 = N3, 18 = N2, 19 = N1. Brains of earlier kinds (1-15, which
read parts of their senses off the track map) are refused (OLD_SENSES).

  Input:  9 wall rays, the sector radar (8 x 4), speed, (wheel), yaw rate, sideslip, health, (battery, tow),
          (N1: pitch, roll, 6 sight lines), heading. All in 0..1, computed by the game (state["obs"]).
  Hidden: 64, ReLU
  Output: steering (left / straight / right) x pedal, or one group per control (action layouts, below)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HIDDEN = 64
# inputs, outputs ("combined" layout) of each class's brain: kinds 16-19
TIERS = {"n4": (46, 3), "n3": (47, 9), "n2": (49, 12), "n1": (57, 12)}
REAL_TIERS = TIERS   # (the same: the senses a real car could measure are the only ones)
SENSES = {"real": TIERS}
GEN_OF_SHAPE = {TIERS["n4"]: 16, TIERS["n3"]: 17, TIERS["n2"]: 18, TIERS["n1"]: 19}   # (with the "combined" layout)
GEN_OF_INPUTS = {shape[0]: g for shape, g in GEN_OF_SHAPE.items()}   # (each input count is one kind)
TIER_OF_GEN = {16: "n4", 17: "n3", 18: "n2", 19: "n1"}
OLD_INPUTS = (6, 18, 19, 21, 22, 24, 31, 38, 39, 42, 43, 44, 45, 51, 52)   # earlier kinds' input counts (refused)
OLD_SENSES = "this brain was made for senses NeuroPrix no longer uses: train a new one"
PEDALS = {"n3": 3, "n2": 4, "n1": 4}                 # pedal choices per tier (N4 has none)
# Action layouts (docs/brain-format.md; backend/sim.py LAYOUTS): how a brain's outputs map onto the car's controls.
# The brain's JSON declares it ("action_layout"); none declared means "combined".
#   combined    one choice over every pedal x steering pair (action = pedal x 3 + steer; Boost is pedal 3)
#   two_head    steering (3) + pedal (N3: gas / coast / brake; N2, N1: + boost), each chosen alone
#   three_head  steering (3) + pedal (gas / coast / brake) + boost (off / on): N2 and N1 only
LAYOUTS = ("combined", "two_head", "three_head")
# A new brain's layout when none is asked for (docs/brain-format.md: three_head measured cleanest for N1). A brain that
# continues keeps its own; other classes train two heads and save them as combined, as always.
DEFAULT_LAYOUT = {"n1": "three_head"}


def layout_outputs(tier: str, layout: str) -> int | None:
    """A class's output count with this layout (None: not allowed for it)."""
    p = PEDALS.get(tier, 0)
    return {"combined": 3 * p if p else 3, "two_head": 3 + p if p else None, "three_head": 8 if p == 4 else None}.get(layout)


def gen_of(n_in: int, n_out: int, layout: str = "combined") -> int:
    """A brain's kind from its inputs, if its outputs fit its layout (0: not a brain)."""
    g = GEN_OF_INPUTS.get(n_in, 0)
    return g if g and layout in LAYOUTS and layout_outputs(TIER_OF_GEN[g], layout) == n_out else 0
STEER = ("left", "straight", "right")
PEDAL = ("accelerate", "coast", "brake", "boost")


class RaceBrain(nn.Module):
    def __init__(self, n_inputs: int = TIERS["n3"][0], n_hidden: int = N_HIDDEN, n_actions: int = TIERS["n3"][1], layout: str = "combined"):
        super().__init__()
        self.layout = layout   # (how its outputs map onto the controls: LAYOUTS)
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> int:
        return int(torch.argmax(self.forward(obs)).item())


def generation(model: "RaceBrain") -> int:
    return gen_of(model.fc1.in_features, model.fc2.out_features, getattr(model, "layout", "combined"))


def tier_of(model: "RaceBrain") -> str:
    return TIER_OF_GEN.get(generation(model), "?")


def state_to_obs(state: dict, gen: int = 17) -> torch.Tensor:
    """Game state JSON -> input tensor (already normalised 0..1 by the game)."""
    feats = state["obs"]
    want = {g: n for n, g in GEN_OF_INPUTS.items()}[gen]
    if len(feats) != want:
        raise ValueError(f"expected {want} features, got {len(feats)}")
    return torch.tensor(feats, dtype=torch.float32).clamp_(0.0, 1.0)


def heads_action(steer: int, pedal: int, boost: int) -> int:
    """The car's action (pedal x 3 + steer, pedal 3 = boost) from three heads' choices (as backend/sim.py)."""
    return (3 if pedal == 0 and boost == 1 else pedal) * 3 + steer


def pick_action(logits: torch.Tensor, layout: str, greedy: bool) -> int:
    """The car's action from a brain's outputs, by its layout: the best of each head (greedy) or each sampled from its
    softmax. Mirrors backend/sim.py Brain.act."""
    def pick(z):
        return int(torch.argmax(z)) if greedy else int(torch.distributions.Categorical(logits=z).sample())
    if layout == "two_head":
        return pick(logits[3:]) * 3 + pick(logits[0:3])
    if layout == "three_head":
        return heads_action(pick(logits[0:3]), pick(logits[3:6]), pick(logits[6:8]))
    return pick(logits)


def relayout(sd: dict, src: str, dst: str, pedals: int) -> dict:
    """A brain's output layer moved from one action layout to another (the inputs and hidden layer kept).
    combined -> two_head: steering and pedal split back out of the joint scores (exact for brains the two-head trainer
    exported as combined: their scores are steer + pedal sums; the closest such split otherwise). two_head ->
    three_head: gas / coast / brake keep their scores, boost on / off get the boost / gas scores, so steering, and
    boost-or-not with the throttle, are exactly as before; only how often the throttle is chosen shifts a little."""
    out = dict(sd)
    for key in ("fc2.weight", "fc2.bias"):
        z = sd[key]
        if src == "combined" and dst in ("two_head", "three_head"):
            M = z.reshape(pedals, 3, *z.shape[1:])   # [pedal, steer, ...]
            z = torch.cat([M.mean(0), M.mean(1) - M.mean((0, 1))])
            src_now = "two_head"
        else:
            src_now = src
        if src_now == "two_head" and dst == "three_head":
            z = torch.cat([z[0:3], z[3:6], z[3:4], z[6:7]])
            src_now = "three_head"
        if src_now != dst:
            raise ValueError(f"can't move a brain from the {src} action layout to {dst}")
        out[key] = z
    return out


def load_layout(path: str | Path) -> str:
    """The action layout a saved brain declares (a .pth: from the .json beside it); "combined" if none."""
    path = Path(path)
    js = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    try:
        layout = json.loads(js.read_text()).get("action_layout") or "combined"
    except (OSError, ValueError, AttributeError):
        return "combined"
    if layout not in LAYOUTS:
        raise ValueError(f"{js.name}: unknown action layout {layout!r}")
    return layout


def save_brain(model: RaceBrain, path: str | Path, name: str = "agent") -> None:
    """Save weights as .pth (for the Grand Prix server) and .json (for the in-browser race viewer)."""
    path = Path(path)
    torch.save(model.state_dict(), path)
    sd = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    brain = {
        "name": name,
        "arch": f"mlp-{model.fc1.in_features}-{model.fc1.out_features}-{model.fc2.out_features}",
        "generation": generation(model),
        "fc1": {"weight": sd["fc1.weight"], "bias": sd["fc1.bias"]},
        "fc2": {"weight": sd["fc2.weight"], "bias": sd["fc2.bias"]},
    }
    if getattr(model, "layout", "combined") != "combined":   # (none declared means combined, as every earlier brain)
        brain["action_layout"] = model.layout
    path.with_suffix(".json").write_text(json.dumps(brain))


# PyTorch before 2.6 can run code hidden in a .pth file even with weights_only=True (CVE-2025-32434)
SAFE_TORCH_LOAD = tuple(int(x) for x in re.findall(r"\d+", torch.__version__)[:2]) >= (2, 6)


def load_weights(path: str | Path) -> dict:
    """A brain's weights (a state_dict) from a .json brain (numbers only: safe with any PyTorch) or a .pth file
    (only with PyTorch 2.6 or newer, which loads weights without running anything hidden in the file)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            b = json.loads(path.read_text())
            return {f"{layer}.{w}": torch.tensor(b[layer][w], dtype=torch.float32) for layer in ("fc1", "fc2") for w in ("weight", "bias")}
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path.name} isn't a brain (.json with fc1 and fc2 weights)") from e
    if not SAFE_TORCH_LOAD:
        raise ValueError(f"PyTorch {torch.__version__} can't load .pth files safely (that needs 2.6 or newer): "
                         f"use the .json brain saved beside it")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"{path.name} isn't a PyTorch state_dict (use the .pth or .json written by train.py)") from e


def load_brain(path: str | Path) -> RaceBrain:
    """Load a brain (.json, or .pth: see load_weights) and check its shapes."""
    sd = load_weights(path)
    if not isinstance(sd, dict):
        raise ValueError("weights file must contain a state_dict")
    expected = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    if set(sd.keys()) != expected:
        raise ValueError(f"unexpected keys {sorted(sd.keys())}; expected {sorted(expected)}")
    hidden, n_in = sd["fc1.weight"].shape
    n_out = sd["fc2.weight"].shape[0]
    layout = load_layout(path)
    if n_in in OLD_INPUTS:
        raise ValueError(OLD_SENSES)
    if not gen_of(n_in, n_out, layout):
        raise ValueError(f"not a NeuroPrix brain: {n_in} inputs and {n_out} outputs don't fit any class with the "
                         f"{layout} action layout")
    if not (1 <= hidden <= 512):
        raise ValueError("hidden layer must have 1..512 neurons")
    model = RaceBrain(n_inputs=n_in, n_hidden=hidden, n_actions=n_out, layout=layout)
    model.load_state_dict(sd)  # raises on any shape mismatch
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("weights contain NaN/inf")
    model.eval()
    return model
