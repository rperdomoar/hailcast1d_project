"""Experimental fixed-mass sphere trajectories; no thermal or growth physics."""
import numpy as np
from .particle_physics import particle_acceleration


def simulate_mechanics(profile, wind_profile, D0_mm, z0, T0_K,
                       x0=0., vx0=0., v0=0., rho_init=900., dt=.1,
                       t_max=60., max_substep_s=.05, updraft_radius_m=None,
                       updraft_center_x_m=0., storm_motion_ms=0., cd=None, drag_law="legacy_piecewise"):
    scalars=[D0_mm,z0,T0_K,x0,vx0,v0,rho_init,dt,t_max,max_substep_s,
             updraft_center_x_m,storm_motion_ms,profile.z_min]
    if not np.all(np.isfinite(scalars)):
        raise ValueError('All scalar inputs must be finite')
    if min(D0_mm,rho_init,dt,t_max,max_substep_s)<=0 or z0<=profile.z_min:
        raise ValueError('Positive size, density and times; start above ground')
    if updraft_radius_m is not None and (not np.isfinite(updraft_radius_m) or updraft_radius_m<=0):
        raise ValueError('Core radius must be positive and finite')
    if cd is not None and (not np.isfinite(cd) or cd<=0):
        raise ValueError('Cd must be positive and finite')
    diameter=D0_mm/1000.; mass=np.pi*diameter**3*rho_init/6
    def air(s,t):
        z=max(float(s[1]),float(profile.z_min))
        factor=1. if updraft_radius_m is None else np.exp(-((s[0]-updraft_center_x_m-storm_motion_ms*t)/updraft_radius_m)**2)
        return np.array([wind_profile.velocity(z),profile.w(z)*factor]),profile.air_density(z),factor
    def rhs(s,t):
        u,rho,_=air(s,t)
        return np.r_[s[2:],particle_acceleration(s[2:],u,mass,diameter,rho,cd=cd,drag_law=drag_law)]
    def step(s,t,h):
        a=rhs(s,t);b=rhs(s+h*a/2,t+h/2);c=rhs(s+h*b/2,t+h/2);d=rhs(s+h*c,t+h)
        out=s+h*(a+2*b+2*c+d)/6
        if not np.all(np.isfinite(out)):raise FloatingPointError('Reduce actual integration substep')
        return out
    state=np.array([x0,z0,vx0,v0],float);time=0.;times=[time];states=[state.copy()]
    steps=[];impacted=False
    while time<t_max:
        outer=min(dt,t_max-time)
        n=max(1,int(np.ceil(outer/max_substep_s)));h=outer/n
        for _ in range(n):
            new=step(state,time,h)
            if new[1]<=profile.z_min:
                lo,hi=0.,h
                for _ in range(40):
                    mid=(lo+hi)/2
                    if step(state,time,mid)[1]>profile.z_min:lo=mid
                    else:hi=mid
                h=hi;new=step(state,time,h);new[1]=profile.z_min;impacted=True
            state=new;time+=h;steps.append(h)
            if impacted:break
        times.append(time);states.append(state.copy())
        if impacted:break
    s=np.array(states);times=np.array(times)
    env=[air(row,t) for row,t in zip(s,times)]
    return {'time':times,'x':s[:,0],'height':s[:,1],'vx':s[:,2],'velocity':s[:,3],
        'mass':np.full(len(s),mass),'radius':np.full(len(s),diameter/2),
        'density':np.full(len(s),rho_init),
        'horizontal_wind':np.array([e[0][0] for e in env]),
        'vertical_air_velocity':np.array([e[0][1] for e in env]),
        'updraft_factor':np.array([e[2] for e in env]),
        'D0_init':D0_mm,'coupling':'experimental_fixed_mass_mechanics',
        'physics_scope':'fixed_mass_fixed_density_no_growth_no_heat_no_melting',
        'drag_law':drag_law,'cd_override':cd,
        'actual_max_step_s':max(steps,default=0.),'event_refined':impacted}
