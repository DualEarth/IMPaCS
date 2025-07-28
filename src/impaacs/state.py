# src/impaacs/state.py

import numpy as np
from .utils import grid_id
from .physics import compute_fractionation, re_bin_sio2
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

    def do_volume_by_layer(
        self,
        n_layers: int,
        sio2_threshold: int
    ) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, int]], int]:
        """
        Summarize SiO2 across layers:
          - percent_volume_by_layer[layer][bin] = % of cells in that bin
          - sum_at_sio2_by_layer[layer][bin] = count of cells in that bin
          - n_cubes_above_threshold = total count across ALL layers for bins>=threshold
        """
        percent_volume_by_layer: dict[int, dict[int, float]] = {}
        sum_at_sio2_by_layer: dict[int, dict[int, int]] = {}

        # gather binned values
        for layer in range(n_layers):
            bar_list: list[int] = []
            for gid, arr in self.grid_state.items():
                val = arr[layer]
                if not np.isnan(val):
                    bar_list.append(re_bin_sio2(val, s_min=1, s_max=100, ds=1))
            # counts and percents
            counts = {b: bar_list.count(b) for b in set(bar_list) if b is not None}
            percents = {b: 100 * counts[b] / len(bar_list) for b in counts}
            sum_at_sio2_by_layer[layer] = counts
            percent_volume_by_layer[layer] = percents

        # total cells above threshold
        n_cubes_above_threshold = sum(
            cnt
            for layer_counts in sum_at_sio2_by_layer.values()
            for b, cnt in layer_counts.items()
            if b >= sio2_threshold
        )

        return percent_volume_by_layer, sum_at_sio2_by_layer, n_cubes_above_threshold