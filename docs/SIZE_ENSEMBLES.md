# Synthetic initial size ensembles

`run_ensemble` now supports `diameter_distribution` or an explicit positive
`diameters_mm` array (one value per member). New sampling options are opt-in;
the legacy clipped-normal sampler is unchanged if neither option is provided.
The legacy `diameter_std_mm` must be zero when either new option is used.

Supported specifications (mm):

- fixed: kind='fixed', value_mm=3
- uniform: kind='uniform', low_mm=1, high_mm=5
- lognormal: kind='lognormal', mean_mm=3, std_mm=1 (arithmetic population moments)
- truncated Gaussian: kind='truncated_normal', location_mm=3, scale_mm=1,
  low_mm=0.1, high_mm=6. Location/scale describe the parent Gaussian, not the
  mean and standard deviation after truncation. Sampling uses inverse-CDF via
  scipy.stats.truncnorm; it does not pile negative draws onto a lower bound.

Each sampled member has equal number weight. These are synthetic inputs, not
observed distributions, measured concentration, mass-weighted distributions,
impacts/m² or a calibrated storm population. Lognormal draws have no finite
upper bound; any resulting out-of-domain Reynolds value is rejected by the
experimental drag law rather than silently clipped.

Distribution sampling uses a SeedSequence child stream, independent of the
legacy position/velocity/wind perturbation stream. Seed and specification are
recorded; explicit samples allow exact paired comparisons among physics modes.
Record NumPy/SciPy versions for bitwise reproducibility across installations.

## Interpretation of the experimental mechanics mode

Fixed mass/density implies fixed diameter throughout each trajectory. It can
change fall time, landing position, speed and kinetic energy, but cannot
simulate an emerging diameter distribution through growth or melting. At a
finite cutoff time the impacted subset may have a different spectrum simply
because slower particles remain aloft. Record the nonimpact fraction and
compare the same member IDs before interpreting spectrum changes.

`python -m audit.run_size_ensembles` runs four distributions with 20 members
each, seed 42, 40 m initial height, 5 m/s uniform horizontal wind, 8 s cutoff,
900 kg/m³ fixed density and opt-in continuous sphere drag. Outputs include
initial diameters, impacted diameters and IDs, nonimpact counts by subtraction,
and maximum per-member diameter change. These are software demonstrations,
not a statistically converged ensemble or observational verification. The
uniform and lognormal populations have nominal mean 3 mm / standard deviation
1 mm; the truncated Gaussian has parent location 3 and scale 1, not exactly
matching truncated moments. Do not call this a shape-only comparison.

Next: nested sample-size convergence and repeated seeds, then explicit
mass/volume and heat/phase-change closure to study generated size spectra.
