"""Complete falls at three heights; uniform air, fixed mass, no phase changes."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile
class Air:
    z_min=0.
    def w(self,z):return 0.
    def air_density(self,z):return 1.2
rows=[]
for dmm in [.9,3.,5.,10.]:
 d=dmm/1000;rho=900.;mass=np.pi*d**3*rho/6;ge=9.81*(1-1.2/rho)
 def force_balance(speed):
    re=1.2*speed*d/1.8e-5
    cd=24/re*(1+.15*re**.687)+.42/(1+42500/re**1.16)
    return 3*1.2*cd/(4*rho*d)*speed**2-ge
 vt=brentq(force_balance,1e-8,80)
 for height in [40.,500.,1000.]:
  pair=[]
  for step in [.1,.05]:
   limit=height/vt+max(20.,5*vt/ge)
   for attempt in range(4):
    r=simulate_trajectory_2d(Air(),HorizontalWindProfile.constant(0),dmm,height,263.15,
       physics_mode='mechanics_reference',drag_law='clift_gauvin',rho_init=rho,
       dt=1.,t_max=limit,max_horizontal_substep_s=step)
    if r['impact']['impacted_ground']:break
    limit*=2
   if not r['impact']['impacted_ground']:raise RuntimeError('Incomplete fall; do not report as impact')
   hit=r['impact'];energy=hit['kinetic_energy_J'];potential=mass*ge*height
   if energy>potential*(1+1e-6):raise AssertionError('Energy exceeds initial effective potential energy')
   np.testing.assert_allclose(r['mass'],mass,rtol=1e-12)
   np.testing.assert_allclose(r['radius'],d/2,rtol=1e-12)
   if abs(hit['vz_ms'])>vt*(1+1e-5):raise AssertionError('Overshoot of terminal speed')
   pair.append(dict(step_s=step,flight_time_s=hit['time_s'],speed_ms=hit['speed_ms'],
       terminal_speed_ms=vt,kinetic_energy_J=energy,effective_potential_J=potential,
       impact_diameter_mm=hit['diameter_mm'],impacted=True,t_max_s=limit))
  coarse,fine=pair
  differences={k:abs(coarse[k]-fine[k])/abs(fine[k]) for k in ['flight_time_s','speed_ms','kinetic_energy_J']}
  if max(differences.values())>1e-3:raise AssertionError('Refine integration: >0.1% difference')
  rows.append(dict(diameter_mm=dmm,height_m=height,runs=pair,relative_step_differences=differences))
 print('completed diameter',dmm,flush=True)
for dmm in [.9,3.,5.,10.]:
 group=[r for r in rows if r['diameter_mm']==dmm]
 assert all(a['runs'][1]['flight_time_s']<b['runs'][1]['flight_time_s'] for a,b in zip(group,group[1:]))
Path('audit/height_consistency.json').write_text(json.dumps(dict(scope='Uniform-air mechanics benchmark; not a cloud/sounding simulation',rows=rows),indent=2)+'\n')
print('maximum relative step difference',max(v for r in rows for v in r['relative_step_differences'].values()))
for dmm in [.9,3,5,10]:
 print(dmm,[(r['height_m'],round(r['runs'][1]['flight_time_s'],3),round(r['runs'][1]['speed_ms'],3)) for r in rows if r['diameter_mm']==dmm])
