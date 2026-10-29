# Experimental integrated mechanics mode

Use `simulate_trajectory_2d(..., physics_mode="mechanics_reference")`.
The default remains `legacy`. The experimental mode fixes each particle's mass,
bulk density and diameter; it has no temperature evolution, accretion, melting,
porosity evolution or phase-change physics. T0_K is accepted for API compatibility
but is not a predicted output. The result states this scope explicitly.

The mode uses the audited vector drag and gravity/buoyancy at every RK4 stage.
`cd=None` uses the existing sphere Reynolds correlation; `cd=0.44` supplies a
constant coefficient for analytical benchmarks, not an empirical hail calibration.
Unsupported growth parameters raise TypeError rather than being ignored.

`dt` controls output intervals. `max_horizontal_substep_s` controls the integration
step of the complete (x,z,vx,vz) state in this mode; `actual_max_step_s` records the
largest step actually used. The inherited parameter name will need a future API
cleanup. Ground crossing is refined by bisection and recomputing the RK4 state;
it does not interpolate diameter and mass independently.

## Verification in this change

21 audit tests pass, including six new integrated-mode tests. All seven existing
trajectory tests pass. The complete sounding-dependent suite was not rerun.
The tests include fixed-Cd analytic velocity and height, fixed geometry, refined
ground crossing, output-interval independence at equal substeps and horizontal
Galilean invariance with the core translated consistently.

`python -m audit.run_integrated_convergence` regenerates 12 runs: 5,10,20 mm;
actual step bounds 0.4,0.2,0.1,0.05 s; output interval fixed at 0.4 s. At 2 s,
for the 5-mm sphere, velocity errors decrease from 3.56e-3 to 1.73e-4, 9.58e-6
and 5.64e-7 m/s. This is consistent with fourth-order convergence in the smooth
fixed-Cd case. See audit/integrated_convergence.json for all results.

This does not establish convergence of the variable-Cd correlation across its
piecewise transitions or of a growing/melting particle. Next: independent
variable-Cd reference integration and size/air-density range checks, then an
explicit mass/volume/thermal closure before publication storm ensembles.
