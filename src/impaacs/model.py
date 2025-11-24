# src/impaacs/model.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ease_grid import EASE2_grid

from .config import IMPAaCSConfig, load_config
from .grid import GridSubsetter
from .state import StateManager
from .physics import impact_dimensions, compute_crust_volume_multiplier
from .utils import grid_id

@dataclass
class ImpactEvent:
    location: Tuple[float, float]
    diameter: float

@dataclass
class SimulationResult:
    times: List[float]
    avg_targets: List[float]
    volume_stats: List[dict[int, float]]
    test_times: List[float]
    test_diams: List[float]
    test_avg_targets: List[float]
    test_top_layers: List[float]

class IMPAaCS:
    def __init__(self, config_path: str):
        # load configuration
        self.config: IMPAaCSConfig = load_config(config_path)

        # initialize grid subsetting
        self.egrid = self._load_ease2_grid()
        subsetter = GridSubsetter(
            egrid=self.egrid,
            lon_limits=self.config.lon_limits,
            lat_limits=self.config.lat_limits,
        )
        self.lon_subset, self.lat_subset = subsetter.subset()

        # compute crust volume multiplier
        self.multiplier: float = compute_crust_volume_multiplier(
            map_scale=self.egrid.map_scale,
            z_discretized_km=self.config.z_discretized_km,
            lon_count=len(self.lon_subset),
            lat_count=len(self.lat_subset),
        )

        # initialize state manager
        self.state = StateManager(
            config=self.config,
            lon_subset=self.lon_subset,
            lat_subset=self.lat_subset,
        )

        # prepare results container
        self.results = SimulationResult(
            times=[],
            avg_targets=[],
            volume_stats=[],
            test_times=[],
            test_diams=[],
            test_avg_targets=[],
            test_top_layers=[],
        )

    def _load_ease2_grid(self):
        return EASE2_grid(self.config.z_discretized_km)

    def update(self, impact: ImpactEvent) -> None:
        """Process a single impact event."""
        # advance sim time
        self.config.sim_time = getattr(impact, 'time', self.config.sim_time)

        # compute geometry
        geom = impact_dimensions(
            diameter=impact.diameter,
            angle_range=self.config.consider_impact_angle,
            max_depth=self.config.max_depth_of_impact_melt_km,
            dz=self.config.z_discretized_km,
        )

        # find impacted cells
        impacted_ids = GridSubsetter(
            egrid=self.egrid,
            lon_limits=self.config.lon_limits,
            lat_limits=self.config.lat_limits,
        ).find_impacted_cells(
            impact_loc=impact.location,
            crater_radius=geom.crater_radius,
            verbose=self.config.verbose,
        )

        if not impacted_ids:
            return

        # compute average target
        avg_tgt = self.state.get_average_target(
            impacted_cells=impacted_ids,
            z_layers=geom.z_layers,
            primitive=self.config.primitive_initial_state,
        )

        # apply impacts
        for gid in impacted_ids:
            self.state.apply_impact(
                gid=gid,
                geom=geom,
                avg_target=avg_tgt,
                config=self.config,
            )

        # record results
        self.results.times.append(self.config.sim_time)
        self.results.avg_targets.append(avg_tgt)
        vol_stats = self.state.do_volume_by_layer(
            n_layers=geom.z_layers,
            sio2_threshold=self.config.sio2_threshold,
        )
        self.results.volume_stats.append(vol_stats)

        # test‐cell tracking
        gid_test = grid_id(*self.config.test_cell_location)
        if gid_test in impacted_ids:
            self.results.test_times.append(self.config.sim_time)
            self.results.test_diams.append(impact.diameter)
            self.results.test_avg_targets.append(avg_tgt)
            top_val = np.mean(self.state.grid_state[gid_test][: self.config.test_layers])
            self.results.test_top_layers.append(top_val)

    def run_sequence(self, impacts: List[ImpactEvent]) -> SimulationResult:
        """Run through a list of impacts in order."""
        for impact in impacts:
            self.update(impact)
        return self.results