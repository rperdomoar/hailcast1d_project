"""Synthetic size spectra and finite-window fallout; no growth validation."""
import json
from pathlib import Path
import numpy as np
from hailcast2d.ensemble import run_ensemble
from hailcast2d.wind_profile import HorizontalWindProfile
class Air:
    z_min=0.
    def w(self,z):return 0.
    def air_density(self,z):return 1.2
specs=[dict(kind='fixed',value_mm=3),dict(kind='uniform',low_mm=3-np.sqrt(3),high_mm=3+np.sqrt(3)),
       dict(kind='lognormal',mean_mm=3,std_mm=1),
       dict(kind='truncated_normal',location_mm=3,scale_mm=1,low_mm=.1,high_mm=6)]
rows=[]
for spec in specs:
 r=run_ensemble(Air(),HorizontalWindProfile.constant(5),20,3,40,263.15,seed=42,
      diameter_distribution=spec,physics_mode='mechanics_reference',drag_law='clift_gauvin',
      rho_init=900,dt=.4,t_max=8,max_horizontal_substep_s=.05)
 errors=[abs(t['radius'][-1]*2000-t['initial_conditions']['D0_mm']) for t in r['trajectories']]
 row=dict(specification=spec,seed=42,n_simulated=r['n_simulated'],n_impacted=r['n_impacted'],
          max_diameter_change_mm=max(errors),
          initial_diameters_mm=[c['D0_mm'] for c in r['initial_conditions']],
          impact_diameters_mm=[c['diameter_mm'] for c in r['impacts']],
          impact_member_ids=[c['member_id'] for c in r['impacts']],
          sampling=r['diameter_sampling'])
 rows.append(row);print(spec['kind'],r['n_impacted'],'/20 impacts; max diameter change',max(errors))
Path('audit/size_ensemble_results.json').write_text(json.dumps(rows,indent=2)+'\n')
