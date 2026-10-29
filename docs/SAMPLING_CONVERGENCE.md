# Sampling convergence pilot

900 trajectories: three seeds (42,123,2026), each with 300 explicit lognormal
initial sizes (population mean 3 mm, std 1 mm). Prefixes 25,50,100,300 are
nested within each seed. Fixed mass/density mechanics, continuous sphere drag,
40 m release height, uniform 5 m/s horizontal wind, 8 s cutoff, 0.05 s maximum
integration step. Each trajectory is computed once. No cloud growth/melting.

At N=300 across three seeds:
- Initial mean diameter: 2.9430 to 3.1024 mm.
- Impact fraction by 8 s: 0.9633 to 0.9833.
- Conditional mean impact energy: 0.0010259 to 0.0014170 J.
- Conditional impact energy p90: 0.0021806 to 0.0031091 J.

These results do NOT establish convergence. Even at 300 members, energy tail
and mean remain seed-sensitive. Mean energy max/min is about 1.38; p90 max/min
about 1.43. More members and independent seeds are needed for precision claims.
Changes across prefixes need not be monotonic. Three seeds provide a pilot
range, not a reliable uncertainty distribution or confidence interval.

Wilson 95% intervals in JSON refer only to binomial impact fraction under this
synthetic independent number-sampling design, not observational uncertainty.
Energy and diameter summaries are conditional on impact before cutoff; members
still aloft must not be interpreted as evaporated or zero-energy hailstones.
Equal number weights imply neither a physical concentration nor energy/m².

Reproduce: python -m audit.run_sampling_convergence
Results: audit/sampling_convergence.json. The prior four-distribution demo and
size sampler are included in this update. Validation: 31 audit tests and four
existing ensemble tests passed. Full sounding-dependent suite not rerun.
