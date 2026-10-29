# HailCast2D publication audit

Reviewed upstream branch: `feature/hailcast2d-crop-risk`.
Reviewed upstream commit: `24a1183503bb99223305e98021cc3602c5bffd3a`.
This directory is a working copy of selected source/data/test files fetched from that commit, not a full upstream checkout. No upstream files or branches have been modified.

## First provisional correction

Both modes now derive initial mass from the requested diameter and initial bulk density. The evolving density state is used consistently to compute geometry in the differential equations and recorded outputs. This removes the inconsistency between the requested diameter, mass, radius, and reported density; it is not a validation of the density closure.

Five audit checks failed on the reviewed source and pass after the correction. All 52 existing tests, including the seven trajectory tests and sounding-dependent modules, pass in the review environment. These are functional and consistency checks, not observational validation. The environment uses the bundled runtime plus locally installed dependencies, not an exact recreation of the pinned requirements.

This changes initial mass and subsequent drag, thermal exchange and collection, so prior results must be regenerated after the physical formulation is settled. Do not mix legacy and corrected simulations.

## Physics requiring a separate decision

1. Vertical acceleration adds `effective_updraft` (m/s) to accelerations (m/s²). The ordinary aerodynamic drag already contains vertical relative velocity. Review the additional memory and lift terms against their intended derivation before removing or replacing them.
2. `porosity` blends ice and water density (900 to 1000 kg/m³); it does not represent air-filled pore volume. Its initialization saturates for rho_init=500 kg/m³. Define the physical state (void fraction versus liquid fraction) and derive its closure and units before calibrating.
3. Coupled horizontal drag uses the full relative-speed norm, while vertical drag and collection use only the vertical relative speed. Derive a common vector force law and define collection geometry.
4. RK4 stage evaluation must maintain physically admissible mass, density and temperature. Small-particle fall-speed clipping differs across coupled/one-way modes; an infinite-width limit is not automatically an exact 1D equivalence.
5. Fusion, freezing and latent heat need a consistent mass/energy budget. `dm=max(dm,-m)` is not a general positivity condition for arbitrary timestep; flooring mass makes `m<=0` termination unreachable in the legacy loop.
6. The sounding case uses 27 January 2021 at 12 UTC, whereas the supplied report describes 00 UTC. Identify the actual source of published figures and whether this sounding represents the storm environment.
7. The label `two_way_finite_updraft` refers to trajectory-growth dependence on a prescribed spatial cloud. It is not two-way particle-to-atmosphere coupling.
8. The ground grid is a 1D transect. Counts of Monte Carlo members and summed sample energies do not establish physical impacts/m² or J/m² without particle weights and an area/time interpretation.

## Ordered experiment plan

| Stage | Controlled comparison | Outputs / decision |
|---|---|---|
| Verification | Still-air terminal fall; co-moving uniform wind; zero LWC; thermal limits | Analytic/independent reference, force/mass/energy residuals |
| Time convergence | Actual integration substeps 0.1, 0.05, 0.025, 0.0125 s, refined as necessary | Diameter, flight time, impact position and energy relative to finest run; provisional <1% acceptance threshold to be justified |
| Mode attribution | Same equations/integrator/initial states in vertical, horizontal one-way, finite-width modes | Attribute differences to geometry rather than numerical or closure changes |
| Environment attribution | Idealized versus sounding-derived profiles with matched initial samples | Separate environmental specification from dimensionality |
| Structural sensitivity | Core radius, storm-relative wind, storm speed, vertical/LWC efficiencies, embryos | Residence time, integrated LWC exposure, diameter/energy distributions and nonimpact fraction |
| Sampling convergence | Nested samples 100, 300, 1000, then increase only if needed; repeated seeds | Confidence intervals on mean and selected quantiles; maximum alone is sample-size dependent |
| Observational evaluation | Event-specific hail observations, with independent cases if available | Sampling-aware comparison of size ranges/distributions and footprint; avoid treating citizen-report counts as hail flux |

The widths and meteorological parameter ranges must be grounded in observations or literature; the listed sample sizes and timesteps are starting designs, not editorial requirements. Keep calibration and evaluation cases separate where possible.
