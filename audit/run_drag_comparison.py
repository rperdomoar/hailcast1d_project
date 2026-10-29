"""Variable-Cd integration checked against independently written DOP853 RHS."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile
class Air:
    z_min=0.
    def __init__(self,rho):self.rho=rho
    def w(self,z):return 0.
    def air_density(self,z):return self.rho
rows=[]
for dmm in [.9,5.,10.,50.]:
 for air in [.4,1.2]:
    d=dmm/1000;rho=900;g=9.81*(1-air/rho)
    def rhs(t,s):
        v=s[1];speed=abs(v)
        if speed==0:return [v,-g]
        re=air*speed*d/1.8e-5
        # Independently implemented expression; does not call code under test.
        cd=24/re*(1+.15*re**.687)+.42/(1+42500/re**1.16)
        return [v,-g-3*air*cd/(4*rho*d)*speed*v]
    ref=solve_ivp(rhs,[0,8],[5000,0],method='DOP853',rtol=1e-11,atol=1e-12,max_step=.02).y[:,-1]
    for step in [.1,.05]:
        r=simulate_trajectory_2d(Air(air),HorizontalWindProfile.constant(0),dmm,5000,263.15,
            physics_mode='mechanics_reference',rho_init=rho,dt=.4,t_max=8,
            max_horizontal_substep_s=step,drag_law='clift_gauvin')
        rows.append(dict(diameter_mm=dmm,air_density_kgm3=air,step_s=step,
             velocity_error_ms=float(abs(r['velocity'][-1]-ref[1])),
             height_error_m=float(abs(r['height'][-1]-ref[0])),
             final_re=float(air*abs(r['velocity'][-1])*d/1.8e-5)))
Path('audit/drag_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print('runs',len(rows),'max velocity error m/s',max(r['velocity_error_ms'] for r in rows),'max height error m',max(r['height_error_m'] for r in rows))
print('finest max velocity error',max(r['velocity_error_ms'] for r in rows if r['step_s']==.05))
