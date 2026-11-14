"""RaceBrain: the small policy network that drives a NeuroPrix car.

Tiers (like the N4 / N3 / N2 ladder; the brain's shape says which it is):
  N4          18 inputs -> 3 actions   steering only, automatic throttle; wall rays + radar + speed
  N3          19 inputs -> 9 actions   steering x pedals (accelerate / coast / brake); + steering wheel
  N2          21 inputs -> 12 actions  N3 + battery and slipstream inputs; pedals accelerate / coast / brake / BOOST
  N4 classic   6 inputs -> 3 actions   the first drivers: wall rays + speed, no radar (still races in N4)
Internally a brain's "generation" is its kind: 1 = N4 classic, 2 = N3, 3 = N4, 4 = N2.

N3 in detail:
  Input  (19): 5 wall rays [L, diag-L, front, diag-R, R]; a radar reporting the 2 nearest cars or
               people (present, ahead, sideways, closing speed, sideways speed, is-a-car); own speed;
               steering-wheel position (the wheel turns at a limited rate, so steering is smooth).
               All in 0..1, computed by the game (state["obs"]).
  Hidden (64): ReLU
  Output (9):  steering (left / straight / right) x pedal (accelerate / coast / brake)
N4 is the same without the wheel input and with 3 steering outputs (the throttle is automatic).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_INPUTS, N_HIDDEN, N_ACTIONS = 19, 64, 9          # N3 (generation 2)
GEN1 = (6, 3)                                        # N4 classic (generation 1): inputs, actions
F3_SHAPE = (18, 3)                                   # N4 (generation 3)
F1_SHAPE = (N_INPUTS + 2, 12)                        # N2 (generation 4): + battery, tow; pedal x4 (boost)
N_CORNER = 3                                         # the corner-ahead sense: track sharpness 30 / 90 / 180 px ahead
F2C_SHAPE = (N_INPUTS + N_CORNER, N_ACTIONS)         # N3 with the corner sense (generation 5)
F1C_SHAPE = (F1_SHAPE[0] + N_CORNER, 12)             # N2 with the corner sense (generation 6)
N1_SHAPE = (F1C_SHAPE[0] + 2 * N_CORNER + 1, 12)     # N1 (generation 7): + grade and bank ahead, the tyres' load
TIERS = {"n4": F3_SHAPE, "n3": F2C_SHAPE, "n2": F1C_SHAPE, "n1": N1_SHAPE}   # what training makes
LEGACY = {"n3": (N_INPUTS, N_ACTIONS), "n2": F1_SHAPE}              # earlier brains of a class (no corner sense): they
                                                                    # still race, and training upgrades them
GEN_OF_SHAPE = {GEN1: 1, (N_INPUTS, N_ACTIONS): 2, F3_SHAPE: 3, F1_SHAPE: 4, F2C_SHAPE: 5, F1C_SHAPE: 6, N1_SHAPE: 7}
TIER_OF_GEN = {1: "n4", 2: "n3", 3: "n4", 4: "n2", 5: "n3", 6: "n2", 7: "n1"}
PEDALS = {"n3": 3, "n2": 4, "n1": 4}                 # pedal choices per tier (N4 has none)
STEER = ("left", "straight", "right")
PEDAL = ("accelerate", "coast", "brake", "boost")


class RaceBrain(nn.Module):
    def __init__(self, n_inputs: int = N_INPUTS, n_hidden: int = N_HIDDEN, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> int:
        return int(torch.argmax(self.forward(obs)).item())


def generation(model: "RaceBrain") -> int:
    return GEN_OF_SHAPE.get((model.fc1.in_features, model.fc2.out_features), 0)


def tier_of(model: "RaceBrain") -> str:
    return TIER_OF_GEN.get(generation(model), "?")


def state_to_obs(state: dict, gen: int = 2) -> torch.Tensor:
    """Game state JSON -> input tensor (already normalised 0..1 by the game)."""
    feats = [float(v) for v in state["sensors"]] + [float(state["speed"])] if gen == 1 else state["obs"]
    want = {1: GEN1[0], 2: N_INPUTS, 3: F3_SHAPE[0], 4: F1_SHAPE[0], 5: F2C_SHAPE[0], 6: F1C_SHAPE[0], 7: N1_SHAPE[0]}[gen]
    if len(feats) != want:
        raise ValueError(f"expected {want} features, got {len(feats)}")
    return torch.tensor(feats, dtype=torch.float32).clamp_(0.0, 1.0)


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
    if (n_in, n_out) not in GEN_OF_SHAPE:
        raise ValueError(f"expected an N3 ({N_INPUTS} inputs, {N_ACTIONS} actions), N4 ({F3_SHAPE[0]}, {F3_SHAPE[1]}) "
                         f"or N4 classic ({GEN1[0]}, {GEN1[1]}) brain, got {n_in} inputs and {n_out} actions")
    if not (1 <= hidden <= 512):
        raise ValueError("hidden layer must have 1..512 neurons")
    model = RaceBrain(n_inputs=n_in, n_hidden=hidden, n_actions=n_out)
    model.load_state_dict(sd)  # raises on any shape mismatch
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("weights contain NaN/inf")
    model.eval()
    return model
