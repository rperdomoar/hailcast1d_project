"""Integrated mechanics mode convergence against an analytic fixed-Cd fall."""
import json
from pathlib import Path
import numpy as np
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile
class Air:
    z_min=0.
    def w(self,z):return 0.
    def air_density(self,z):return 1.2
rows=[]
for diameter in [5.,10.,20.]:
    g=9.81*(1-1.2/900);vt=np.sqrt(4*(diameter/1000)*9.81*(900-1.2)/(3*.44*1.2))
    exact_v=-vt*np.tanh(g*2/vt);exact_z=5000-vt**2/g*np.log(np.cosh(g*2/vt))
    for step in [.4,.2,.1,.05]:
        r=simulate_trajectory_2d(Air(),HorizontalWindProfile.constant(0),diameter,5000,263.15,
            physics_mode='mechanics_reference',rho_init=900,dt=.4,t_max=2,
            max_horizontal_substep_s=step,cd=.44)
        rows.append(dict(diameter_mm=diameter,requested_max_step_s=step,actual_max_step_s=r['actual_max_step_s'],
                         velocity_error_ms=float(abs(r['velocity'][-1]-exact_v)),
                         height_error_m=float(abs(r['height'][-1]-exact_z))))
Path('audit/integrated_convergence.json').write_text(json.dumps(rows,indent=2)+'\n')
for diameter in [5.,10.,20.]:
    a=[r for r in rows if r['diameter_mm']==diameter]
    print(diameter,[(r['actual_max_step_s'],r['velocity_error_ms'],r['height_error_m']) for r in a])
