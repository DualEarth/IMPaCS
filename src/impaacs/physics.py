# src/impaacs/physics.py

import random
import numpy as np
from .constants import CURRENT_CRUST_VOLUME_KM3, SURFACE_AREA_EARTH_KM2
from dataclasses import dataclass

@dataclass
class ImpactGeometry:
    crater_radius: float
    penetration_depth: float
    z_layers: int


def impact_dimensions(diameter: float, angle_range: tuple[float,float], 
                      max_depth: float, dz: float) -> ImpactGeometry:
    angle_factor = random.uniform(*angle_range)
    crater_radius = 5 * diameter  # radius = (10*diameter)/2
    penetration_depth = min(max_depth, angle_factor * diameter)
    z_layers = int(penetration_depth / dz)
    return ImpactGeometry(crater_radius, penetration_depth, z_layers)

def compute_fractionation(target: float, upper: float) -> float:
    return 1 - (target / upper)

def re_bin_sio2(value: float, s_min: int, s_max: int, ds: int) -> int:
    if value < s_min:
        return s_min
    if value > s_max:
        return s_max
    bins = list(range(s_min, s_max + ds, ds))
    return min(b for b in bins if value <= b)