# Brain format

A driver's brain is a small neural network with a fixed shape per class:

```
inputs (what the car senses, 0..1)  ->  64 hidden units (ReLU)  ->  scores (logits), grouped by its action layout
```

The table shows the `combined` layout (one score per action); see [action layouts](#action-layouts) for the others.

| Class | Inputs | Hidden | Actions | Brain kind (`generation`) |
|---|---|---|---|---|
| N4 | 46 | 64 | 3 | 16 |
| N3 | 47 | 64 | 9 | 17 |
| N2 | 49 | 64 | 12 | 18 |
| N1 | 57 | 64 | 12 | 19 |

It's sent as JSON (`brain_out`, and the starting brain in a session):

```json
{
  "name": "Dex Jango",
  "arch": "mlp-45-64-12",
  "generation": 14,
  "fc1": {"weight": [[...45 numbers...], ... 64 rows], "bias": [...64 numbers...]},
  "fc2": {"weight": [[...64 numbers...], ... 12 rows], "bias": [...12 numbers...]}
}
```

The car's decision: `h = relu(fc1.weight · x + fc1.bias)`, `scores = fc2.weight · h + fc2.bias`. In races the driver
usually takes its highest-scoring action, and now and then one it scores nearly as highly (softmax of the scores
divided by 0.2), so the same drivers on the same circuit don't race identically every time. With heads (below), it
does that in each head on its own.

The rules:

- `fc1.weight` is 64 rows of *inputs* numbers, `fc2.weight` is *actions* rows of 64 numbers; the shapes must match
  the driver's class, or the server refuses the brain.
- Only numbers: brains are data. The server never runs anything from a brain.
- Only the JSON is uploaded. The server keeps the weights, the driver's own name and `action_layout`; any other field
  is dropped.
  (A `.pth` next to it, PyTorch's `torch.save` of a state dict, is for your own use: it's never sent.)
- How you train is up to you, but the network must be this one: you can't add layers or units. (A class may get a
  new brain shape in a future season; the SDK and the server will say so.)

Brains of earlier kinds (1–15, with other input counts: senses partly read off the track map) are refused, by the
engine, the trainer and the server: "this brain was made for senses NeuroPrix no longer uses: train a new one". Start
a new brain instead.

## Action layouts

Each decision the car gets three controls: **steering** (left, straight, right), **pedal** (gas, coast, brake; N4 has
no pedal) and, in N2 and N1, **boost** (off, on; it only does anything with gas). A brain's `action_layout` says how its
outputs become those controls. Leave it out and it's `combined`.

| `action_layout` | Outputs | How the car picks | Classes |
|---|---|---|---|
| `combined` | one per steering x pedal pair: N4 3, N3 9, N2 / N1 12 | the best of all of them: action = pedal x 3 + steering, pedal 3 = gas with boost | all |
| `two_head` | 3 steering, then the pedals: N3 3 + 3 = 6, N2 / N1 3 + 4 = 7 (boost as a 4th pedal) | the best steering and, separately, the best pedal | N3, N2, N1 |
| `three_head` | 3 steering, 3 pedal (gas, coast, brake), 2 boost (off, on) = 8 | the best of each of the three on its own | N2, N1 |
| `proportional` | 7 steering (full, two thirds, a third left; straight; a third, two thirds, full right), 5 pedal (full brake, half brake, coast, half gas, full gas), N2 / N1 2 boost (off, on): N3 12, N2 / N1 14 | the best of each on its own: action = 1000 + steering + 7 x pedal + 35 x boost | N3, N2, N1 |

```json
{"name": "Nora Summit", "arch": "mlp-57-64-14", "generation": 19, "action_layout": "proportional", "fc1": {...}, "fc2": {...}}
```

- **New N3, N2 and N1 brains are `proportional`** by default (engine 2026.11.9): the cars handle like real ones (see
  [the engine](engine.md#how-the-car-handles)), and a brain that steers part-way and brakes or accelerates by half
  catches slides that all-or-nothing controls turn into spins. Before, new N1 brains were `three_head`. The server says
  which layout a session wants (`settings.layout`, [the trainer contract](trainer-contract.md)); your brain must have
  the layout it declares, and the output count must fit it and the class, or the server refuses it.
- Only these four fixed layouts exist. The server never builds any other network: there's no field for layers,
  sizes or code.
- In Python, `neuroprix.brain` has the shims: `LAYOUTS`, `DEFAULT_LAYOUT`, `layout_outputs(tier, layout)`,
  `pick_action(scores, layout, greedy)` (scores to the car's action), `relayout(state_dict, src, dst, pedals)` (move a
  brain forward: combined -> two_head -> three_head, keeping what it learned; nothing moves to proportional) and
  `load_layout(path)`. The engine's `neuroprix.engine.heads_action(steer, pedal, boost)` turns three choices into the
  action number, `neuroprix.engine.controls(action)` any action into a steering target and throttle and brake levels, and
  `neuroprix.engine.Brain` reads any layout. `neuroprix-train --layout ...` trains any of them.
