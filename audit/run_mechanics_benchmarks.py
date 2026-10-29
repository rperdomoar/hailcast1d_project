"""Run from repo root with python -m audit.run_mechanics_benchmarks."""
import json
from pathlib import Path
import numpy as np
from hailcast2d.particle_physics import particle_acceleration, dry_void_fraction
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile


def run():
    rows=[]
    for d in [.005,.01,.02]:
        rho,air,cd,g=900.,1.2,.44,9.81
        mass=np.pi*d**3*rho/6
        effective_g=g*(1-air/rho)
        vt=np.sqrt(4*d*g*(rho-air)/(3*cd*air))
        target=-vt*np.tanh(effective_g*10/vt)
        for dt in [.2,.1,.05,.025]:
            v=np.zeros(2)
            def f(velocity):return particle_acceleration(velocity,[0,0],mass,d,air,cd=cd)
            for _ in range(round(10/dt)):
                k1=f(v);k2=f(v+dt*k1/2);k3=f(v+dt*k2/2);k4=f(v+dt*k3)
                v+=dt*(k1+2*k2+2*k3+k4)/6
            rows.append({'diameter_mm':d*1000,'dt_s':dt,'time_s':10,
                         'numerical_vz_ms':float(v[1]),'analytic_vz_ms':float(target),
                         'relative_velocity_error':float(abs((v[1]-target)/target))})
    class UniformCloud:
        z_min=0.
        def w(self,z):return 5.
        def lwc(self,z):return 0.
        def temperature(self,z):return 263.15
        def air_density(self,z):return 1.2
    legacy=[]
    for w in [0.,5.]:
        cloud=UniformCloud();cloud.w=lambda z,w=w:w
        h=1e-6
        r=simulate_trajectory_2d(cloud,HorizontalWindProfile.constant(3),5,5000,263.15,
             v0=w,vx0=3,t_max=h,dt=h,updraft_radius_m=3000,k_wind=0)
        estimated=(r['velocity'][1]-r['velocity'][0])/h
        legacy.append({'air_vertical_ms':w,'particle_initial_vertical_ms':w,
                       'initial_vz_acceleration_estimate_ms2':float(estimated),
                       'gravity_only_reference_ms2':-9.81})
    out={'scope':'Independent fixed-mass, fixed-density, constant-Cd mechanics benchmark; not storm validation',
         'constant_drag_fall':rows,'legacy_comoving_probe':legacy,
         'dry_void_fraction_for_500kgm3_ice900':dry_void_fraction(500)}
    Path('audit/mechanics_benchmarks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'largest_fall_error':max(x['relative_velocity_error'] for x in rows),
      'finest_step_largest_error':max(x['relative_velocity_error'] for x in rows if x['dt_s']==.025),
      'legacy_comoving_probe':legacy,'porosity':out['dry_void_fraction_for_500kgm3_ice900']},indent=2))
if __name__=='__main__':run()
