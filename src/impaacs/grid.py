# src/impaacs/grid.py
from typing import List, Tuple
import numpy as np
from ease_grid import EASE2_grid
from .utils import distance, grid_id

class GridSubsetter:
    def __init__(self, egrid: EASE2_grid, lon_limits: tuple, lat_limits: tuple):
        self.egrid = egrid
        self.lon_limits = lon_limits
        self.lat_limits = lat_limits
        self.lon_subset: list[float] = []
        self.lat_subset: list[float] = []
        self._compute_subsets()

    def _compute_subsets(self) -> None:
        for lon in self.egrid.londim:
            if self.lon_limits[0] < lon < self.lon_limits[1]:
                self.lon_subset.append(lon)
        for lat in self.egrid.latdim:
            if self.lat_limits[0] < lat < self.lat_limits[1]:
                self.lat_subset.append(lat)

    def subset(self) -> tuple[list[float], list[float]]:
        return self.lon_subset, self.lat_subset
    
    def find_impacted_cells(
        self,
        impact_loc: Tuple[float,float],
        crater_radius: float,
        verbose: bool = False,
    ) -> List[str]:
        """
        Return list of grid_ids hit by an impact at `impact_loc`
        with the given `crater_radius`. If none fall within the radius,
        assign to the closest cell when it’s within one grid-spacing.
        """
        impacted: List[str] = []
        dmin = np.inf

        # first pass: collect all within radius, track min distance
        for lon in self.lon_subset:
            for lat in self.lat_subset:
                d = distance(impact_loc[0], lat, impact_loc[1], lon)
                dmin = min(dmin, d)
                if d <= crater_radius:
                    impacted.append(grid_id(lon, lat))

        # fallback: if nothing hit but the closest is within one grid cell (~30 km)
        if not impacted and dmin < 30:
            if verbose:
                print(f"Warning: no cells within {crater_radius} km, closest is {dmin:.1f} km")
            # find the exact lon/lat of that min distance
            for lon in self.lon_subset:
                for lat in self.lat_subset:
                    if distance(impact_loc[0], lat, impact_loc[1], lon) == dmin:
                        impacted.append(grid_id(lon, lat))
                        if verbose:
                            print(f"Assigning to closest grid {impacted[-1]}")
                        break
                if impacted:
                    break

        return impacted