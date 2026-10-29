"""Nested synthetic lognormal ensembles; fixed-mass mechanics, not validation."""
import json
from pathlib import Path
import numpy as np
from hailcast2d.ensemble import run_ensemble
from hailcast2d.size_distributions import sample_diameters
from hailcast2d.wind_profile import HorizontalWindProfile
class Air:
    z_min=0.
    def w(self,z):return 0.
    def air_density(self,z):return 1.2
rows=[]
for seed in [42,123,2026]:
    sizes=sample_diameters(300,dict(kind='lognormal',mean_mm=3,std_mm=1),seed)
    r=run_ensemble(Air(),HorizontalWindProfile.constant(5),300,3,40,263.15,
       diameters_mm=sizes,seed=seed,physics_mode='mechanics_reference',drag_law='clift_gauvin',
       rho_init=900,dt=.4,t_max=8,max_horizontal_substep_s=.05)
    for n in [25,50,100,300]:
        hits=[h for h in r['impacts'] if h['member_id']<n];p=len(hits)/n;z=1.96
        den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        ds=np.array([h['diameter_mm'] for h in hits]);es=np.array([h['kinetic_energy_J'] for h in hits])
        rows.append(dict(seed=seed,n=n,initial_mean_mm=float(sizes[:n].mean()),
            initial_p90_mm=float(np.quantile(sizes[:n],.9)),n_impacted=len(hits),impact_fraction=p,
            impact_fraction_wilson95=[center-half,center+half],
            impact_mean_mm=float(ds.mean()),impact_p90_mm=float(np.quantile(ds,.9)),
            impact_mean_energy_J=float(es.mean()),impact_p90_energy_J=float(np.quantile(es,.9))))
    print('finished seed',seed,flush=True)
summary=[]
for n in [25,50,100,300]:
    group=[r for r in rows if r['n']==n];s=dict(n=n)
    for k in ['initial_mean_mm','impact_fraction','impact_mean_energy_J','impact_p90_energy_J']:
        vs=[r[k] for r in group];s[k]=dict(mean=float(np.mean(vs)),min=min(vs),max=max(vs))
    summary.append(s)
out=dict(scope='Synthetic sampling convergence only; fixed mass/density, finite 8-s cutoff; conditional impact statistics',
         specification=dict(kind='lognormal',mean_mm=3,std_mm=1),seeds=[42,123,2026],
         nested_prefixes=[25,50,100,300],rows=rows,across_seed_summary=summary)
Path('audit/sampling_convergence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
