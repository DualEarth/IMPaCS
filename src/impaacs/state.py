# src/impaacs/state.py

import numpy as np
from .utils import grid_id
from .physics import compute_fractionation
from .utils import clip_to_bounds

class StateManager:
    def __init__(self, config, lon_subset, lat_subset):
        self.config = config
        self.lon_subset = lon_subset
        self.lat_subset = lat_subset
        self.grid_state = {}
        self._state_prep()

    def _state_prep(self):
        total_layers = int(self.config.max_depth_of_impact_melt_km / self.config.z_discretized_km)
        for lon in self.lon_subset:
            for lat in self.lat_subset:
                gid = grid_id(lon, lat)
                self.grid_state[gid] = np.full(total_layers, self.config.primitive_initial_state)

    def get_average_target(self, impacted_cells: list[str], z_layers: int, primitive: float) -> float:
        total = 0
        for gid in impacted_cells:
            arr = self.grid_state.get(gid, np.full(z_layers, primitive))
            total += arr[:z_layers].sum()
        return total / (len(impacted_cells) * z_layers)

    def apply_impact(self, gid: str, geom: ImpactGeometry, avg_target: float, config):
        upper_n = int(ceil(config.fraction_upper_layer * geom.z_layers))
        lower_n = geom.z_layers - upper_n
        fractionation = compute_fractionation(config.target_SiO2, config.upper_SiO2)

        # primitive mantle
        for i in range(geom.z_layers, geom.max_layers):
            self.grid_state[gid][i] = config.primitive_initial_state

        # upper melt
        wt_upper = avg_target / (1 - fractionation)
        for i in range(upper_n):
            val = wt_upper
            if config.bound_sio2:
                val = clip_to_bounds(val, config.sio2_bounds)
            self.grid_state[gid][i] = round(val, 1)

        # lower melt
        numerator = avg_target - (config.fraction_upper_layer * wt_upper)
        for i in range(upper_n, geom.z_layers):
            val = numerator / config.fraction_lower_layer
            if config.bound_sio2:
                val = clip_to_bounds(val, config.sio2_bounds)
            self.grid_state[gid][i] = round(val, 1)