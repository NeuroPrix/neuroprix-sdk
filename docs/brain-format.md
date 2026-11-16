# Brain format

A driver's brain is a small neural network with a fixed shape per class:

```
inputs (what the car senses, 0..1)  ->  64 hidden units (ReLU)  ->  one score (logit) per action
```

| Class | Inputs | Hidden | Actions | Brain kind (`generation`) |
|---|---|---|---|---|
| N4 | 18 | 64 | 3 | 3 |
| N3 | 22 | 64 | 9 | 5 |
| N2 | 24 | 64 | 12 | 6 |
| N1 | 31 | 64 | 12 | 7 |

It's sent as JSON (`brain_out`, and the starting brain in a session):

```json
{
  "name": "Dex Jango",
  "arch": "mlp-24-64-12",
  "generation": 6,
  "fc1": {"weight": [[...24 numbers...], ... 64 rows], "bias": [...64 numbers...]},
  "fc2": {"weight": [[...64 numbers...], ... 12 rows], "bias": [...12 numbers...]}
}
```

The car's decision: `h = relu(fc1.weight · x + fc1.bias)`, `scores = fc2.weight · h + fc2.bias`. In races the driver
usually takes its highest-scoring action, and now and then one it scores nearly as highly (softmax of the scores
divided by 0.2), so the same drivers on the same circuit don't race identically every time.

The rules:

- `fc1.weight` is 64 rows of *inputs* numbers, `fc2.weight` is *actions* rows of 64 numbers; the shapes must match
  the driver's class, or the server refuses the brain.
- Only numbers: brains are data. The server never runs anything from a brain.
- Only the JSON is uploaded. The server keeps the weights and the driver's own name; any other field is dropped.
  (A `.pth` next to it, PyTorch's `torch.save` of a state dict, is for your own use: it's never sent.)
- How you train is up to you, but the network must be this one: you can't add layers or units. (A class may get a
  new brain shape in a future season; the SDK and the server will say so.)

Older brains of a class (kind 2 for N3, 4 for N2: before the corner-ahead sense) still race; the default trainer
upgrades them when it continues them (the new inputs start switched off).

The sector-radar generation (see [what a driver senses](observations-and-actions.md#the-sector-radar-the-next-generation-of-brain))
has 20 more inputs in each class: N4 38 inputs (kind 8), N3 42 (kind 9), N2 44 (kind 10), N1 51 (kind 11), with the same
hidden layer and actions. A class races one kind at a time: the server accepts the sector-radar kind for a class once
that class has moved to it.
