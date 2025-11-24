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
    max_layers: int


def impact_dimensions(
    diameter: float,
    angle_range: tuple[float,float],
    max_depth: float,
    dz: float
) -> ImpactGeometry:
    angle_factor = random.uniform(*angle_range)
    crater_radius = 5 * diameter  # (10×diameter)/2
    penetration_depth = min(max_depth, angle_factor * diameter)
    z_layers = int(penetration_depth / dz)
    # this is the *absolute* max number of layers you ever have:
    max_layers = int(max_depth / dz)
    return ImpactGeometry(crater_radius, penetration_depth, z_layers, max_layers)

def compute_fractionation(target: float, upper: float) -> float:
    return 1 - (target / upper)

def re_bin_sio2(value: float, s_min: int, s_max: int, ds: int) -> int:
    if value < s_min:
        return s_min
    if value > s_max:
        return s_max
    bins = list(range(s_min, s_max + ds, ds))
    return min(b for b in bins if value <= b)

def compute_crust_volume_multiplier(
    map_scale: float,
    z_discretized_km: float,
    lon_count: int,
    lat_count: int,
    surface_area_earth_km2: float = SURFACE_AREA_EARTH_KM2,
    total_crust_km3: float = CURRENT_CRUST_VOLUME_KM3,
) -> float:
    """
    Compute the relative percent crustal volume multiplier:
    (grid_area * sample_area_ratio) / total_crust.
    """
    # grid cell area in km²
    grid_area = (map_scale / 1000.0) ** 2
    # thickness of each cube
    cube_vol = grid_area * z_discretized_km
    # number of cubes in the sample (one layer)
    n_cubes = lon_count * lat_count
    sample_area = grid_area * n_cubes
    sample_ratio = sample_area / surface_area_earth_km2
    # scale cube volume up to whole‐Earth
    cube_vol_scaled = cube_vol / sample_ratio
    # convert to relative percent
    return cube_vol_scaled / total_crust_km3