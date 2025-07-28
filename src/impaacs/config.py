# src/impaacs/config.py

from dataclasses import dataclass
import yaml

@dataclass
class IMPAaCSConfig:
    verbose: bool
    ensemble: int
    sim_time: float
    max_depth_of_impact_melt_km: float
    lon_limits: tuple[float, float]
    lat_limits: tuple[float, float]
    z_discretized_km: float
    primitive_initial_state: float
    fraction_upper_layer: float
    target_SiO2: float
    upper_SiO2: float
    proportion_melt_from_impact: float
    n_layers_impact_melt: int
    bound_sio2: bool
    sio2_bounds: tuple[int, int]
    consider_impact_angle: tuple[float, float]
    sio2_threshold: int
    test_layers: int


def load_config(path: str) -> IMPAaCSConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    # flatten nested dicts if needed
    return IMPAaCSConfig(
        verbose = data['verbose'],
        ensemble = data['ensemble'],
        sim_time = data['simulation']['sim_time'],
        max_depth_of_impact_melt_km = data['simulation']['max_depth_of_impact_melt_km'],
        lon_limits = tuple(data['grid']['lon_limits']),
        lat_limits = tuple(data['grid']['lat_limits']),
        z_discretized_km = data['grid']['z_discretized_km'],
        primitive_initial_state = data['model']['primitive_initial_state'],
        fraction_upper_layer = data['model']['fraction_upper_layer'],
        target_SiO2 = data['model']['target_SiO2'],
        upper_SiO2 = data['model']['upper_SiO2'],
        proportion_melt_from_impact = data['model']['proportion_melt_from_impact'],
        n_layers_impact_melt = data['model']['n_layers_impact_melt'],
        bound_sio2 = data['model']['bound_sio2'],
        sio2_bounds = tuple(data['model']['sio2_bounds'].values()),
        consider_impact_angle = tuple(data['impact']['consider_impact_angle']),
        sio2_threshold = data['thresholds']['sio2_threshold'],
        test_layers = data['test']['test_layers'],
    )
