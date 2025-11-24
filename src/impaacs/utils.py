# src/impaacs/utils.py

from math import radians, sin, cos, sqrt, atan2
from .constants import EARTH_RADIUS_KM, SIO2_BOUNDS

def distance(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Haversine distance in kilometers"""
    φ1, φ2 = radians(lat1), radians(lat2)
    λ1, λ2 = radians(lon1), radians(lon2)
    Δφ, Δλ = φ2 - φ1, λ2 - λ1
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def grid_id(lon: float, lat: float, prec: int = 4) -> str:
    return f"{lon:.{prec}f} {lat:.{prec}f}"

def clip_to_bounds(value: float, bounds: tuple[float,float] = SIO2_BOUNDS) -> float:
    """Clamp a value to the given min/max bounds."""
    return max(bounds[0], min(bounds[1], value))