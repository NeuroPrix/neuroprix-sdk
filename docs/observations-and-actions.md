# What a driver senses and can do

The car decides 15 times a second: each decision lasts 4 physics frames (60 frames a second). Every input is 0..1.

## Inputs, in order

These are today's brains (the 2-slot radar, kinds 3-7), which every class races now. The next generation swaps
inputs 6-17 for [the sector radar](#the-sector-radar-the-next-generation-of-brain), which sees all round the car.

| # | Input | Classes | Meaning |
|---|---|---|---|
| 1–5 | Wall rays | all | Distance to the edge of the track at −90°, −45°, 0°, +45°, +90° from the nose (left to right), ÷ 250 px |
| 6–11 | Radar slot 1 | all | The nearest car ahead or alongside within 200 px: present (1/0), distance ahead, sideways offset (0.5 = in line), closing speed (0.5 = same speed), sideways speed (0.5 = none), is-a-car (1). Empty: `0, 1, 0.5, 0.5, 0.5, 0` |
| 12–17 | Radar slot 2 | all | The second nearest, the same way |
| 18 | Speed | all | ÷ top speed (5 px a frame) |
| 19 | Steering wheel | N3, N2, N1 | 0 = full left, 0.5 = straight, 1 = full right (the wheel turns at a limited rate) |
| 20 | Battery | N2, N1 | Charge left for boost |
| 21 | Tow | N2, N1 | Slipstream from a car just ahead (1 = right behind it) |
| 22–24 (N3: 20–22) | Corners ahead | N3, N2, N1 | The track's sharpness 30, 90 and 180 px ahead: 0.5 = straight, above 0.5 a right-hand bend, below a left-hand one; 0 or 1 = a hairpin |
| 25–27 | Grade ahead | N1 | The road's slope 30, 90 and 180 px ahead: 0.5 = level, 1 = 15% or more uphill, 0 = 15% or more downhill |
| 28–30 | Bank ahead | N1 | The road's bank at those points: 0.5 = flat, above 0.5 banked into the corner (more grip), below off-camber (less), 0/1 = 20° or more |
| 31 | Load | N1 | How heavy the car is on its tyres now: 0.5 = normal, less over a crest (light: less grip, traction and braking), more in a dip |

N4 has inputs 1–18. N3 has 1–19 and the corners (22). N2 has 1–24. N1 has all 31.

## The sector radar (the next generation of brain)

The engine also has a second radar, the brain kinds 8–11 (N4 38 inputs, N3 42, N2 44, N1 51). A class races one
kind at a time, so every driver in a race senses the same things: a class moves to the sector radar on a set date,
announced in advance; until then its races use the radar above. You can train sector-radar brains already
(`neuroprix-train --senses sectors`; `--resume` with one of today's brains moves it across).

It replaces inputs 6–17 with 32 inputs: the circle round the car is cut into 8 sectors, and each reports the nearest
car whose centre is in it, within 200 px, behind as well as ahead. Everything else keeps its order, 20 places later.

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

A wrecked car reads as a car standing still. Obstacles aren't on the radar (stones stop the wall rays, as before), so
the "is a car" input is gone.

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
