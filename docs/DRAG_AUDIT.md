# Drag audit

## Findings

The legacy piecewise sphere coefficient is discontinuous at Re=1 and Re=1000.
One-sided coefficient limits are 24 to 27.4 (+14.17%) and 0.60792058 to 0.44
(-27.62%). At fixed positive geometry and fluid properties these are also
relative jumps in force magnitude. They are artifacts of the selected formula,
not evidence of a physical hail drag crisis at these Reynolds numbers.

The experimental mechanics mode now accepts `drag_law="clift_gauvin"`:

    Cd = 24/Re (1 + 0.15 Re^0.687) + 0.42/(1 + 42500/Re^1.16)

This is an opt-in smooth-sphere reference. The default `legacy_piecewise` is
unchanged, as are production legacy trajectories. A positive constant `cd`
overrides the coefficient for analytical checks. The experimental candidate is
restricted conservatively to 0 < Re <= 2e5 and rejects larger values. At zero
slip drag is returned as exactly zero without evaluating singular Cd. The
vector force uses the full relative-speed norm in both components.

Source for candidate equation and its incompressible continuum pre-critical
scope: A general drag coefficient model for a spherical particle incorporating
rarefaction and particle-to-gas temperature ratio effects, Chemical Engineering
Science (2024), https://www.sciencedirect.com/science/article/pii/S0009250924007425 .
The indexed primary-source excerpt supplies the formula; full text retrieval
was blocked. This audit does not establish its accuracy for natural hail.

Natural hail can be nonspherical, rough, tumbling or melting. An applicable
hail-specific empirical parameterization and its valid size/Re range remain
necessary before claiming forecast skill. The new branch is intended to
separate numerical continuity from empirical aerodynamic validation.

## Completed verification

26 audit tests and seven existing trajectory tests pass. Added five tests for
Stokes asymptote, continuity at the old thresholds, unchanged default,
dissipation and invalid/out-of-range inputs.

`python -m audit.run_drag_comparison` compares integrated variable-Cd falls with
an independently written ODE right-hand side using DOP853 at tight tolerances.
Sixteen runs cover diameters 0.9,5,10,50 mm, air densities 0.4,1.2 kg/m³ and
actual substep bounds 0.1,0.05 s. The reference uses the same physical
correlation, so this is a numerical cross-check, not an alternative empirical
validation. Mass and density are fixed, with zero wind and no growth/heat.
At 8 s maximum velocity difference is 1.94e-7 m/s, maximum height difference
1.35e-4 m. For 0.05-s runs maximum velocity difference is 1.84e-8 m/s.
See audit/drag_comparison.json for individual results.

Full sounding-dependent suite was not rerun. Next tests should exercise
horizontal wind shear and vector force coupling, viscosity/air-temperature
sensitivity, relaxation of very small embryos, and comparison with hail-specific
fall-speed/drag observations. Current checks do not validate energy at impact.
