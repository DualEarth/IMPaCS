# src/impaacs/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .utils import grid_id, clip_to_bounds


def plot_map_and_bar(
    grid_state: dict[str, np.ndarray],
    lon_subset: list[float],
    lat_subset: list[float],
    percent_volume_by_layer: dict[int, dict[int, float]],
    sim_time: float,
    fig_path: str = './',
    save_figure: bool = False,
    plot_figure: bool = False,
    map_layers: list[int] = [0],
    dist_layer: int = 0,
    bound_plots: bool = True,
    lon_limits: tuple[float,float] = None,
    lat_limits: tuple[float,float] = None,
) -> None:
    """
    Plot a 2D contour map of mean SiO2 over specified layers and a bar chart
    of SiO2 distribution for a given layer.

    Parameters:
    - grid_state: mapping grid_id -> 1D array of SiO2 values per layer
    - lon_subset, lat_subset: coordinate lists defining the sample grid
    - percent_volume_by_layer: distro for bar chart (layer -> {bin: percent})
    - sim_time: simulation time in years
    - fig_path: output directory/prefix for saved figure
    - save_figure, plot_figure: toggles
    - map_layers: which layers to average for map
    - dist_layer: which layer for bar chart
    - bound_plots: whether to enforce axis/color bounds
    - lon_limits, lat_limits: map extent (required if bound_plots)
    """
    if not save_figure and not plot_figure:
        print('not plotting figure')
        return

    # build data matrix for contour
    nx, ny = len(lon_subset), len(lat_subset)
    z = np.zeros((nx, ny))
    for i, lon in enumerate(lon_subset):
        for j, lat in enumerate(lat_subset):
            gid = grid_id(lon, lat)
            vals = [grid_state[gid][layer] for layer in map_layers]
            mean_val = np.mean(vals)
            z[i, j] = clip_to_bounds(mean_val) if bound_plots else mean_val

    X, Y = np.meshgrid(lon_subset, lat_subset)
    fig = plt.figure(figsize=(12, 7))
    gs = plt.GridSpec(1, 7, wspace=0.1, hspace=0.1)

    # contour map
    ax_map = fig.add_subplot(gs[0, :5])
    if bound_plots:
        levels = np.arange(40, 70, 2)
    else:
        levels = None
    cs = ax_map.contourf(
        X, Y, z.T,
        levels=levels,
        cmap=cm.get_cmap('jet', (len(levels) - 1) if levels is not None else 256)
    )
    cbar = fig.colorbar(cs, ax=ax_map, ticks=range(40,70,2) if bound_plots else None)
    ax_map.set_title(f'Surface SiO2 at {int(sim_time/1e6)} Myr')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    if bound_plots and lon_limits and lat_limits:
        ax_map.set_xlim(lon_limits)
        ax_map.set_ylim(lat_limits)
        ax_map.set_xticks(np.arange(lon_limits[0], lon_limits[1]+1, 10))

    # bar chart
    ax_bar = fig.add_subplot(gs[0, 5:])
    distro = percent_volume_by_layer.get(dist_layer, {})
    bins = sorted(distro.keys())
    percents = [distro[b] for b in bins]
    ax_bar.bar(bins, percents, width=1.2)
    if bound_plots:
        ax_bar.set_xlim(40, 70)
        ax_bar.set_ylim(0, 50)
        ax_bar.set_xticks(np.arange(40, 71, 5))
    ax_bar.set_xlabel('SiO2 content (wt%)')
    ax_bar.set_ylabel(f'Percent volume layer {dist_layer}')

    # save/show
    if save_figure:
        out = f"{fig_path}{int(sim_time/1e6)}Myr.png"
        fig.savefig(out, bbox_inches='tight', dpi=100)
    if plot_figure:
        plt.show()
    plt.close(fig)