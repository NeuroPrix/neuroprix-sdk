# What a driver senses and can do

The car decides 15 times a second: each decision lasts 4 physics frames (60 frames a second). Every input is 0..1.

## Inputs, in order

New brains sense only what a real car or its driver could measure, plus which way the race goes (brain kinds 16–19:
N4 46 inputs, N3 47, N2 49, N1 57; engine 2026.11.6). Nothing is read off the track map but the heading.

| # | Input | Classes | Meaning |
|---|---|---|---|
| 1–9 | Wall rays | all | Distance from the nose to the first point that isn't road (the track's edge or a rock) at −90°, −60°, −40°, −20°, 0°, +20°, +40°, +60°, +90° from the nose (left to right), ÷ 250 px. The 0° and ±20° rays are [beams the car's width](#the-forward-beams) |
| 10–41 | Radar | all | 8 sectors all round the car, 4 numbers each: [the radar](#the-radar-8-sectors-all-round) below |
| 42 | Speed | all | ÷ top speed (5 px a frame) |
| 43 | Steering wheel | N3, N2, N1 | 0 = full left, 0.5 = straight, 1 = full right (the wheel turns at a limited rate) |
| next | Yaw rate | all | How fast the body is turning, over the last frame: 0.5 = not turning, above 0.5 = turning right; ±3 rad/s reads 0 or 1 (the tyres hold about 2.3 rad/s at top speed, so a spin reads 0 or 1) |
| next | Sideslip | all | The angle from the nose to the way the car is moving: 0.5 = going where it points, above 0.5 = moving to the right of the nose; ±90° reads 0 or 1. 0.5 when nearly stopped. N4 can't slide, so it reads 0.5 |
| next | Health | all | 1 = undamaged. Damage cuts the car's top speed to (0.5 + 0.5 × health) of its best until a pit stop |
| next | Battery | N2, N1 | Charge left for boost |
| next | Tow | N2, N1 | Slipstream from a car just ahead (1 = right behind it): you feel it as less drag |
| next | Pitch | N1 | The car's tilt along its length: 0.5 = level, above = nose up; ±0.15 rad reads 0 or 1 |
| next | Roll | N1 | The car's tilt across: 0.5 = level, above = its right side lower; ±0.3 rad reads 0 or 1 |
| next 3 | Sight lines, left | N1 | [What the eye sees](#sight-lines-n1) of the left road edge 30, 90 and 180 px ahead |
| next 3 | Sight lines, right | N1 | The same on the right |
| last | Heading | all | Which way the track runs against the car's nose: 0.5 = straight along it, below 0.5 the track runs to the left (0.25: 90° left), above to the right (0.75: 90° right), 0 or 1 = facing backwards |

So N4 has the rays, the radar, speed, yaw rate, sideslip, health and the heading (46). N3 adds the wheel after speed
(47). N2 adds the battery and tow after health (49). N1 adds pitch, roll and the six sight lines before the heading
(57).

## The forward beams

The 0° and ±20° rays are as wide as the car (10 px): each is three parallel rays, from the middle of the nose and from
5 px either side, and the shortest reading counts. A rock just off the nose's line, which a single ray would pass, shows
up as the car closes on it. The others are single rays: across the road, a gap the width of a car doesn't matter.

## Sight lines (N1)

Each sight line starts at a front corner of the car (left or right), on the road, and ends on that side's road edge
30, 90 or 180 px further round the circuit. Its reading is the line's angle above or below the car's own level (the
plane of the road under the car): 0.5 + angle ÷ full scale, the full scales being ±0.05, ±0.10 and ±0.15 rad for the
three distances. On a steady slope or a steady bank every line reads 0.5, so the lines show how the road changes ahead
(a crest, a dip, a corner banked more or less), while pitch and roll give the slope now. When a crest hides the end, the
line reads the highest point on the way (the brow), which is what a driver sees. On a flat track every sight line,
pitch and roll read 0.5.

The road's surface: a height along the middle of the road and a tilt across it in banked corners, the outside edge
higher. Banks only exist where the road curves: a track's bank fades out as a bend straightens and is zero on a
straight, whatever a track file says.

## Earlier brains

Earlier brains still race, each sensing what it was built for, and `neuroprix-train` keeps a continuing one in its own
senses (an earlier brain's inputs mean other things, so it never moves to the ones above):

- **Kinds 12–15** (N4 39 inputs, N3 43, N2 45, N1 52): 5 wall rays at −90°, −45°, 0°, +45°, +90° (the forward three
  beams), the radar (6–37), speed (38), the wheel (39, N3 up), battery and tow (40–41, N2 and N1), the track's
  sharpness 30, 90 and 180 px ahead read off the map (N3 up: 0.5 straight, above a right-hand bend), N1's grade and
  bank at those points and its tyres' load, then the heading, last.
- **Kinds 8–11** (N4 38, N3 42, N2 44, N1 51): the same without the heading. `neuroprix-train` adds it when it
  continues one (`neuroprix.brain.add_heading`: one more input with zero weights, so it drives exactly as before until
  it learns to use it), and the [minimal example](../examples/minimal_trainer.py) shows the same in NumPy.

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

### The 2-slot radar, in the earliest brains

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
off-camber one less; banking only counts in corners (it fades out as a bend straightens). Flat tracks drive exactly as
they always have.

Walls and stones cost health by how hard you hit them (a square-on hit at top speed wrecks the car); touching another
car costs both half their speed. Too fast for a corner and the car drifts wide. A wrecked car, one that makes no
progress for 10 seconds, or one going the wrong way is out. The full rules are the engine itself
(`neuroprix.engine`): it's the same code the official races run.
