"""Number-weighted synthetic initial diameters in mm, not observed hail spectra."""
import numpy as np
from scipy.stats import truncnorm


def sample_diameters(n, specification, seed=None):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0:
        raise ValueError('n must be a positive integer')
    spec=dict(specification);kind=spec.pop('kind',None);rng=np.random.default_rng(seed)
    allowed={'fixed':{'value_mm'},'uniform':{'low_mm','high_mm'},
             'lognormal':{'mean_mm','std_mm'},
             'truncated_normal':{'location_mm','scale_mm','low_mm','high_mm'}}
    if kind not in allowed or set(spec)!=allowed[kind]:
        raise ValueError('Unknown distribution or incorrect parameter names')
    if not all(np.isfinite(float(v)) for v in spec.values()):raise ValueError('Finite parameters required')
    if kind=='fixed':
        if spec['value_mm']<=0:raise ValueError('Positive diameter required')
        values=np.full(n,spec['value_mm'])
    elif kind=='uniform':
        lo,hi=spec['low_mm'],spec['high_mm']
        if not 0<lo<hi:raise ValueError('Require 0 < low < high')
        values=rng.uniform(lo,hi,n)
    elif kind=='lognormal':
        mean,std=spec['mean_mm'],spec['std_mm']
        if mean<=0 or std<0:raise ValueError('Positive mean and nonnegative std required')
        variance_log=np.log1p((std/mean)**2)
        values=rng.lognormal(np.log(mean)-variance_log/2,np.sqrt(variance_log),n)
    else:
        loc,scale,lo,hi=(spec[k] for k in ['location_mm','scale_mm','low_mm','high_mm'])
        if scale<=0 or not 0<lo<hi:raise ValueError('Positive scale and 0 < low < high required')
        values=truncnorm.rvs((lo-loc)/scale,(hi-loc)/scale,loc=loc,scale=scale,size=n,random_state=rng)
    values=np.asarray(values,float)
    if not np.all(np.isfinite(values)) or np.any(values<=0):raise ValueError('Invalid sampled diameters')
    return values
