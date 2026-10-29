"""Audited spherical-particle reference physics, not yet wired to production.

Fixed instantaneous mass/geometry; no variable-mass momentum loading, heat,
melting, liquid retention or history forces. Up is positive z. SI throughout.
See docs/PHYSICS_AUDIT_02.md for provenance and limitations.
"""
import numpy as np


def _positive(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _vector(name, value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite (x,z) vector')
    return value


def sphere_drag_coefficient(reynolds):
    """Legacy sphere correlation; zero slip is handled before evaluating Cd."""
    re = _positive('reynolds', reynolds)
    if re < 1:
        return 24/re
    if re < 1000:
        return 24/re + 6/(1 + np.sqrt(re)) + .4
    return .44


def drag_acceleration(velocity, air_velocity, mass_kg, diameter_m,
                      air_density_kgm3, viscosity_pas=1.8e-5, cd=None):
    v, u = _vector('velocity', velocity), _vector('air_velocity', air_velocity)
    mass = _positive('mass', mass_kg)
    diameter = _positive('diameter', diameter_m)
    rho = _positive('air_density', air_density_kgm3)
    mu = _positive('viscosity', viscosity_pas)
    if cd is not None:
        cd = _positive('cd', cd)
    slip = v-u
    speed = float(np.linalg.norm(slip))
    if speed == 0:
        return np.zeros(2)
    coefficient = sphere_drag_coefficient(rho*speed*diameter/mu) if cd is None else cd
    area = np.pi*diameter**2/4
    return -.5*coefficient*rho*area/mass * speed*slip


def particle_acceleration(velocity, air_velocity, mass_kg, diameter_m,
                          air_density_kgm3, gravity_ms2=9.81,
                          include_buoyancy=True, **kwargs):
    gravity = _positive('gravity', gravity_ms2)
    acceleration = drag_acceleration(velocity, air_velocity, mass_kg,
                                     diameter_m, air_density_kgm3, **kwargs)
    volume = np.pi*diameter_m**3/6
    buoyancy = air_density_kgm3*volume/mass_kg if include_buoyancy else 0.
    acceleration[1] -= gravity*(1-buoyancy)
    return acceleration


def collection_rate(lwc_kgm3, efficiency, diameter_m, velocity, air_velocity):
    """Swept-volume interception; assumes droplets co-move with air.

    This is a collection kernel, not an unconditional frozen-mass growth rate.
    """
    diameter = _positive('diameter', diameter_m)
    lwc, eff = float(lwc_kgm3), float(efficiency)
    if not np.isfinite(lwc) or lwc < 0 or not np.isfinite(eff) or not 0 <= eff <= 1:
        raise ValueError('Require LWC >= 0 and 0 <= efficiency <= 1')
    slip = _vector('velocity', velocity)-_vector('air_velocity', air_velocity)
    return float(eff*np.pi*diameter**2/4*lwc*np.linalg.norm(slip))


def dry_bulk_density(void_fraction, ice_density_kgm3=900.):
    """Ice plus massless voids only; not a wet-growth density closure."""
    fraction = float(void_fraction)
    rho = _positive('ice_density', ice_density_kgm3)
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('void_fraction must satisfy 0 <= p < 1')
    return (1-fraction)*rho


def dry_void_fraction(bulk_density_kgm3, ice_density_kgm3=900.):
    bulk = _positive('bulk_density', bulk_density_kgm3)
    ice = _positive('ice_density', ice_density_kgm3)
    if bulk > ice:
        raise ValueError('Dry bulk density cannot exceed solid ice density')
    return 1-bulk/ice
