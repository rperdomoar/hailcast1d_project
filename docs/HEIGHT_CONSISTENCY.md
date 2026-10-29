# Complete-fall height consistency

Run `python -m audit.run_height_consistency` from repository root.
24 trajectories: four diameters (0.9,3,5,10 mm), three heights (40,500,1000 m),
two actual integration step bounds (0.1,0.05 s). Identical uniform air density
1.2 kg/m³, zero wind/updraft, fixed particle density 900 kg/m³, continuous sphere
drag, release from rest. No growth, thermal evolution, melting or real sounding.
Initial temperature is unused by this experimental mechanics mode.

A separate scalar force-balance calculation estimates terminal speed and sets
an initial time limit. If no impact occurs, the runner doubles the limit, up to
four attempts; uncompleted falls raise an error instead of entering impact
statistics. Ground crossing is refined by the experimental integrator.

All 24 runs reached ground. Diameter and mass stayed constant. Time to impact
increased with initial height. Speed did not overshoot independently calculated
terminal speed. In still air, impact kinetic energy did not exceed initial
effective gravitational potential energy (including buoyancy).

Finest-step flight times (s):

| Diameter mm | 40 m | 500 m | 1000 m |
|---|---:|---:|---:|
| 0.9 | 12.417 | 152.225 | 304.190 |
| 3 | 5.377 | 60.006 | 119.386 |
| 5 | 4.352 | 45.099 | 89.388 |
| 10 | 3.662 | 33.715 | 66.370 |

Maximum relative difference between the two steps across flight time, impact
speed and kinetic energy: 2.93e-6 (0.000293%). The runner rejects differences
above a provisional 0.1% numerical threshold. This threshold is an engineering
check, not a journal requirement or uncertainty on natural hail.

The 0.9-mm particle takes longer than 8 s even from 40 m, illustrating the old
finite-window censoring. Uniform density is intentionally held constant across
all heights to isolate geometric fall-distance effects. Next, test a prescribed
height-varying air-density profile and wind shear with an independent integrator;
thermal/mass closure is required before interpreting cloud-release spectra.

Validation this session: 31 audit tests and seven existing trajectory tests
passed. Full sounding-dependent suite not rerun. Results are stored in
audit/height_consistency.json. No observational validation is claimed.
