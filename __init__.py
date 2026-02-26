# -*- coding: utf-8 -*-
"""
Paquete hailcast1d: Simulación de granizo 1D
API pública del modelo
"""

from .hailcast1d import (
    run_deterministic_simulation,
    run_case,
    run_updraft_set,
    run_all_cases
)

from .hail_growth_stochastic import run_montecarlo
from .cloud_model import CloudProfile
from .hail_animation import animate_hailstone, animate_hail_growth
from .hail_analysis import analyze_all_profiles

__all__ = [
    "run_deterministic_simulation",
    "run_case",
    "run_updraft_set",
    "run_all_cases",
    "run_montecarlo",
    "CloudProfile",
    "animate_hailstone",
    "animate_hail_growth",
    "analyze_all_profiles",
]
