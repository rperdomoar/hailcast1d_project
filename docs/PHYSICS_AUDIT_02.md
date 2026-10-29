# Second audit — dynamics and density

## Outcome

The production equations still require reformulation. Added an independent, opt-in reference module `hailcast2d/particle_physics.py`; no production import or default simulation behavior was changed in this audit. This lets us test a proposed mechanical reference without treating legacy storm outputs as targets.

## 1. Mechanical inconsistency reproduced

With a spatially uniform air velocity, initialize the particle co-moving with the air. Relative velocity is initially zero. In a gravity-and-drag approximation, initial aerodynamic drag must vanish. Buoyancy, if included, slightly reduces gravity, independent of uniform translation.

The legacy coupled integrator gives these initial accelerations, measured over 1e-6 s (5-mm initial sphere, bulk density 500 kg/m³, no LWC or erosion):

| Vertical air / particle speed | Estimated initial vertical acceleration |
|---|---:|
| 0 / 0 m/s | -9.8100 m/s² |
| 5 / 5 m/s | -5.8333 m/s² |

The code contains additional `effective_updraft` and `lift` terms. The first has velocity units but is added to acceleration. The second depends on absolute vertical air speed rather than particle slip. Neither is part of the tested gravity-plus-vector-drag reference. These are numerical measurements of the local implementation, not observed hail accelerations.

Reference mechanics, with upward z and slip s = v_hail - v_air:

    a_drag = -rho_air Cd A |s| s / (2m)
    a_z,gravity+buoyancy = -g (1 - rho_air V/m)

The same slip magnitude must enter both drag components. No separate wind-velocity acceleration or unsigned w² lift is added in the reference. It neglects variable-mass momentum transfer, pressure-gradient, added-mass, history and shape effects. Before coupling to accretion, explicitly assess momentum carried by collected droplets rather than assuming the fixed-mass reference is a complete growth model.

The module also supplies a spherical interception kernel Ec A LWC |s| assuming droplets co-move with air. This does not specify freezing fraction, water retention, shedding or thermal balance and must not be equated unconditionally with ice-mass growth.

## 2. Porosity is not liquid fraction

For dry ice with negligible void-air mass, void fraction p satisfies rho_bulk = (1-p) rho_ice. With rho_ice=900 and rho_bulk=500 kg/m³, p=0.44444. A decrease in void fraction raises density.

The production formula `(1-p)*rho_ice + p*rho_water` instead resembles a two-condensed-phase volume mixture. At p=0.9 it gives 990 kg/m³. Its inverse saturates for the default initial density of 500 kg/m³. Furthermore its supposed compaction reduces p and consequently reduces the target density, opposite to dry compaction.

Added dry density/void conversion functions as diagnostic references, not a replacement wet-growth closure. Do not switch production porosity to these formulas while retaining the old dp/dt, empirical relaxation and melting rules.

Recommended controlled baseline: fixed bulk density within each benchmark, then a separate mass/volume evolution model for dry accretion with explicit deposition density. Wet growth needs distinct ice, liquid and void-volume accounting, plus an energy/freezing/retention model. Changing the existing 0.3 s^-1 density relaxation alone has no demonstrated physical basis.

## 3. Verification completed

- 15 audit tests pass: five previous geometry tests and ten new mechanics/collection/porosity tests.
- Seven existing production trajectory tests pass.
- Zero-slip drag, dissipative relative drag, Galilean invariance, rotational covariance, gravity/buoyancy limit, fixed-Cd terminal balance, collection and input domains were checked.
- Twelve independent falling-sphere integrations (5, 10, 20 mm; dt 0.2, 0.1, 0.05, 0.025 s) were compared at 10 s with vz(t) = -vt tanh(g_effective t/vt), for constant mass/density/Cd. Largest relative error: 1.54e-8; largest at the finest timestep: 3.30e-12.
- This exact-solution benchmark does not validate a variable-Cd trajectory, phase change, a sounding, or the full storm model.
- The full 52-test sounding-dependent suite was not rerun in this turn: temporary dependencies from the previous environment no longer exist. No full-suite pass is claimed for this audit.

Run from repository root:

    python -m unittest discover -s audit -v
    python -m unittest discover -s tests -p test_trajectory_2d.py -v
    python -m audit.run_mechanics_benchmarks

The last command regenerates `audit/mechanics_benchmarks.json`.

## 4. References and status

Adams-Selin and Ziegler (2016), Forecasting Hail Using a One-Dimensional Hail Growth Model within WRF. DOI: https://doi.org/10.1175/MWR-D-16-0027.1 . The published HAILCAST uses terminal velocity relative to updraft and physically distinct wet/dry growth treatments; our inertial reference is not a reproduction of that algorithm.

Kumjian, Spychalla and Lombardo (2025), Hailstone Inertial Adjustment to Storm Wind Fields and Implications for Growth, ECSS conference poster: https://presentations.copernicus.org/ECSS2025/ECSS2025-207_supplement.pdf . This primary conference contribution explicitly gives vector drag/gravity and a relative-speed collection kernel. It is not treated as a peer-reviewed journal validation.

Kumjian and Lombardo (2020), A Hail Growth Trajectory Model for Exploring the Environmental Controls on Hail Size: Model Physics and Idealized Tests. DOI: https://doi.org/10.1175/JAS-D-20-0016.1 . Relevant for separately developing and testing dry/wet growth and volume changes.

## Next implementation block

Integrate the reference mechanics behind an explicit experimental selector, keep legacy mode for comparison, use fixed density for controlled mechanical benchmarks, add the same reference to 1D and 2D, and verify the infinite-width and zero-horizontal-slip limits. Resolve the thermal/mass/volume closure before running publication ensembles or fitting parameters to the current 7.8-mm result.
