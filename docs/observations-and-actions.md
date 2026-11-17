# What a driver senses and can do

The car decides 15 times a second: each decision lasts 4 physics frames (60 frames a second). Every input is 0..1.

## Inputs, in order

Every class senses with the sector radar and the heading sense (brain kinds 12–15: N4 39 inputs, N3 43, N2 45, N1 52).

| # | Input | Classes | Meaning |
|---|---|---|---|
| 1–5 | Wall rays | all | Distance to the edge of the track or a rock at −90°, −45°, 0°, +45°, +90° from the nose (left to right), ÷ 250 px. The forward three (−45°, 0°, +45°) are [beams the car's width](#the-forward-beams) |
| 6–37 | Radar | all | 8 sectors all round the car, 4 numbers each: [the radar](#the-radar-8-sectors-all-round) below |
| 38 | Speed | all | ÷ top speed (5 px a frame) |
| 39 | Steering wheel | N3, N2, N1 | 0 = full left, 0.5 = straight, 1 = full right (the wheel turns at a limited rate) |
| 40 | Battery | N2, N1 | Charge left for boost |
| 41 | Tow | N2, N1 | Slipstream from a car just ahead (1 = right behind it) |
| 42–44 (N3: 40–42) | Corners ahead | N3, N2, N1 | The track's sharpness 30, 90 and 180 px ahead: 0.5 = straight, above 0.5 a right-hand bend, below a left-hand one; 0 or 1 = a hairpin |
| 45–47 | Grade ahead | N1 | The road's slope 30, 90 and 180 px ahead: 0.5 = level, 1 = 15% or more uphill, 0 = 15% or more downhill |
| 48–50 | Bank ahead | N1 | The road's bank at those points: 0.5 = flat, above 0.5 banked into the corner (more grip), below off-camber (less), 0/1 = 20° or more |
| 51 | Load | N1 | How heavy the car is on its tyres now: 0.5 = normal, less over a crest (light: less grip, traction and braking), more in a dip |
| last (N4 39, N3 43, N2 45, N1 52) | Heading | all | Which way the track runs against the car's nose: 0.5 = straight along it, below 0.5 the track runs to the left (0.25: 90° left), above to the right (0.75: 90° right), 0 or 1 = facing backwards |

N4 has inputs 1–38 and the heading (39). N3 has 1–39, the corners (40–42) and the heading (43). N2 has 1–44 and the
heading (45). N1 has all 52.

## The forward beams

The three forward wall rays (−45°, 0° and +45°) are as wide as the car (10 px): each is three parallel rays, from the
middle of the nose and from 5 px either side, and the shortest reading counts. A rock just off the nose's line, which a
single ray would pass, shows up as the car closes on it. The side rays (−90°, +90°) are single rays. Every brain,
whatever its kind, senses the beams (engine 2026.11.5).

## The heading sense

The last input says which way the circuit runs compared with where the car is pointing: the angle from the nose to the
direction of the track under the car, as 0.5 + angle ÷ 360°. A car spun round by a hit can tell it's facing the wrong
way (near 0 or 1) and which way to turn. Brains of kinds 8–11 (the sector radar without it: N4 38 inputs, N3 42,
N2 44, N1 51) still race as they were built; `neuroprix-train` adds the heading when it continues one
(`neuroprix.brain.add_heading`: one more input with zero weights, so it drives exactly as before until it learns to use
it), and the [minimal example](../examples/minimal_trainer.py) shows the same in NumPy.

## The radar: 8 sectors all round

The circle round the car is cut into 8 sectors, and each reports the nearest car whose centre is in it, within
200 px, behind as well as ahead.

| Sector | Bearing from the nose (+ = right) |
|---|---|
| front | −15° to +15° |
| front-left / front-right | 15° to 60° either side |
| left / right | 60° to 120° either side |
| rear-left / rear-right | 120° to 160° either side |
| rear | the last 40°, behind |

In that order (front, front-left, left, rear-left, rear, rear-right, right, front-right), 4 numbers each:

| Input | Meaning | Empty sector |
|---|---|---|
| seen | 1 if a car is in the sector | 0 |
| near | 1 − distance ÷ 200 px (centre to centre): 1 = touching | 0 |
| closing | 0.5 + how fast the gap is shrinking ÷ 10 px a frame: above 0.5 = getting closer, whether it's catching you or you're catching it | 0.5 |
| sideways | 0.5 + its sideways speed relative to you ÷ 10 px a frame: above 0.5 = moving to your right | 0.5 |

A wrecked car reads as a car standing still. Obstacles aren't on the radar (stones stop the wall rays).

### Earlier brains: the 2-slot radar

Brains made before the sector radar (kinds 3–7: N4 18 inputs, N3 22, N2 24, N1 31) have a radar on the 2 nearest
cars ahead or alongside instead of inputs 6–37: 12 inputs, 6 per car (present, distance ahead, sideways offset,
closing speed, sideways speed, and "is a car", always 1; an empty slot reads 0, 1, 0.5, 0.5, 0.5, 0). Everything
else is the same, 20 places earlier, without the heading. They still race, each sensing what it was built for.
`neuroprix-train` moves one to the sector radar and the heading sense when it continues it
(`neuroprix.brain.to_sectors`: the old radar's inputs dropped, the sectors starting switched off; then
`add_heading`), and the [minimal example](../examples/minimal_trainer.py) shows the same in NumPy.

## Actions

- **N4** (3): 0 = left, 1 = straight, 2 = right. The throttle is automatic.
- **N3** (9), **N2** and **N1** (12): action = pedal × 3 + steering, with steering 0 = left, 1 = straight, 2 = right, and
  pedal 0 = accelerate, 1 = coast (lift off), 2 = brake, and in N2 and N1 3 = boost (uses the battery; with nothing to
  give it's the throttle).

That's the action the engine takes (`env.step_world`). A brain's outputs can be laid out three ways, one score per
action (`combined`) or one group per control (`two_head`, `three_head`): see
[action layouts](brain-format.md#action-layouts). New N1 brains are `three_head`.

## The rules of the road, briefly

**Hills (N1's circuits and any track with heights or banks):** uphill lowers the car's top speed (a 10% climb: about
−25%) and downhill raises it (up to +15%); gravity pulls along the slope, so coasting downhill gains speed and braking
downhill takes longer. Over a crest the car goes light (less grip, throttle and brakes, by how sharp the crest is and how
fast it's going); in a dip it's pressed down (more grip). A corner banked into the turn holds more sideways grip, an
off-camber one less. Flat tracks drive exactly as they always have.

Walls and stones cost health by how hard you hit them (a square-on hit at top speed wrecks the car); touching another
car costs both half their speed. Too fast for a corner and the car drifts wide. A wrecked car, one that makes no
progress for 10 seconds, or one going the wrong way is out. The full rules are the engine itself
(`neuroprix.engine`): it's the same code the official races run.
