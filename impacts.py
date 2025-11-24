#!/home/jmframe/programs/anaconda3/bin/python3

import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from ease_grid import EASE2_grid
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
import random
import pickle as pkl

# approximate radius of earth in km
def distance(lat1,lat2,lon1,lon2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return(distance) #km

class IMPAaCS:

    """
    Update: June 3rd 2021 @ 11:30AM Central Time, Setting limits for SiO2 percent bins from 1-100, so we can get min/max
    Update: May 29th 2021 @ 2:30PM Central Time, Search only the subset of grids for impact, else skip
    Update: May 28th 2021 @ 12:19PM Central Time, with Jordan to get ensembles running
    Update: December 15th 2021 @ 3:30 Pacific time, putting bounds (40-80) on SiO2 
    Update: January 27th 2022 @ 11PM Pacific time, Saving the maximum sio2 at each of the top 12 vertical layers
    Update: Jamuary 30th 2022 @ 11:45PM Pacific time, adding option to bound sio2
    Update: February 1-13th 2022 Adding capability to do SiO2 percent volumes BY LAYER
    Update: February 15th 2022 @ Adding factor from 1-3 to the impact deth to accound for impactor angle.
                                 Adding a sum_at_sio2_by_layer to calculate pure volume, instead of percent.
    Update: February 21st 2022: can test average of n layers at the "test cell". Simplifying the vertical discretization of impacts a bit.
    Update: March 4th 2022: The impact_diameter was not used in the get_average_target function, so I removed it.
    Update: March 9th 2022: Fixed the calculation for wt_sio2_upper. Add back in the angle factor and make sure it works correctly
    Update: March 10th 2022: Angle factor range as input
    Dynamic geospatial model of IMPaCS, 
    using the size-frequency distribution of impacts scaled from the lunar surface, 
    we generate the volume and abundance of this enriched crust on Earth’s surface 
    during the Hadean to determine how rapidly it evolved.
    """
    
    impact_test_id = str(round(0.18672199,4))+' '+str(round(0.14122179,4))
    
    def __init__(self, egrid, 
                 verbose=False,
                 max_depth_of_impact_melt=500,
                 ensemble = 0,
                 primitive_initial_state=45,
                 fraction_upper_layer = 2/3,
                 target_SiO2 = 62.58,     # From sudbury
                 upper_SiO2 = 68.71,      # From sudbury
                 n_layers_impact_melt = 2,
                 z_discretized_km = int(2),
                 proportion_melt_from_impact = 1/3,
                 sim_time=0,
                 lon_lims = [-180, 180], lat_lims = [-45, 45],
                 bound_sio2=False,
                 test_layers=1,
                 consider_impact_angle=[1,3],
                 sio2_threshold=58):
        self.egrid = egrid
        self.verbose=verbose
        self.ensemble=ensemble
        self.primitive_initial_state = primitive_initial_state
        self.max_depth_of_impact_melt = max_depth_of_impact_melt
        self.fraction_upper_layer = fraction_upper_layer        # d_upper / Mi (from Sudbury)
        self.fraction_lower_layer = 1-self.fraction_upper_layer # d_lower / Mi (from Sudbury)
        self.n_layers_impact_melt = n_layers_impact_melt
        self.target_SiO2 = target_SiO2 # From sudbury
        self.upper_SiO2 = upper_SiO2  # From sudbury
        self.z_discretized_km = z_discretized_km
        self.proportion_melt_from_impact = proportion_melt_from_impact
        self.average_target = self.primitive_initial_state
        self.average_target_list = [self.primitive_initial_state]
        self.top_layers_at_test_cell = [self.primitive_initial_state]
        self.average_test_target_list = [self.primitive_initial_state]
        self.n_x = self.egrid.londim.shape[0]
        self.n_y = self.egrid.latdim.shape[0]
        self.sim_time=sim_time
        self.bound_sio2=bound_sio2
        self.count_test_hits = 0
        self.grid_cell_state = {}
        self.impacted_grid_cells = []
        self.impactors_at_test_cell = [0]
        self.test_time = [0]
        self.test_layers = test_layers
        self.sio2_threshold = sio2_threshold
        self.sum_at_sio2_by_layer = {}
        self.percent_volume_by_layer = {}
        self.relative_percent_crust_vol_list = []
        self.n_cubes_above_threshold_list = []
        self.relative_percent_crust_vol_multiplier = np.nan
        self.lon_lims=lon_lims
        self.lat_lims=lat_lims
        self.lon_subset=[]
        self.lat_subset=[]

        self.consider_impact_angle = consider_impact_angle
        self.impact_angle_factor = 1
        self.penetration_depth = 0
        
        # Set up the grids so we don't search the whole planet for every impact
        self.get_subset_of_grids()
        self.n_x = len(self.lon_subset)
        self.n_y = len(self.lat_subset)
        
        # This has to go after subsetting the n_x and n_y, because there is a function of sample area
        self.calculate_relative_percent_crust_vol_multiplier()
        
        # Finally we set up the state data
        self.state_prep()
        
    #--------------------------------------------------------------------------------------------------
    def update(self, impact_loc, impactor_diameter, sim_time=0):
        self.sim_time = sim_time
        self.impact_dimensions(impactor_diameter)
        self.find_the_grid(impact_loc)
        if len(self.impacted_grid_cells) > 0:
            self.get_average_target()
            self.loop_impact_grid(impactor_diameter)
        
    #--------------------------------------------------------------------------------------------------
    #---- THIS IS THE MAIN CODE -------------------- THIS IS THE MAIN CODE ----------------------------
    #--------------------------------------------------------------------------------------------------
    def state_dynamics(self, impactor_diameter, grid_cell_id):
        """
        This is the critical component of this model
        This function will change the chemical makeup of each grid cell
            according to the chemical theory put forward by Faltys-Wielicki [2021]
        """
        
        #####      DYNAMIC FACTORS       ############################
        depth_of_impact_melt = self.impact_angle_factor * impactor_diameter * self.proportion_melt_from_impact # D/3

        #Vertical discretization.
        melt_layers = int(np.ceil(depth_of_impact_melt / self.z_discretized_km))
        
        n_upper_layers = int(np.ceil(self.fraction_upper_layer * melt_layers))

        if self.verbose:
            print("depth_of_impact_melt", depth_of_impact_melt)
            print("melt_layers", melt_layers)
            print("n_upper_layers", n_upper_layers)

        upper_layer  = range(0, n_upper_layers)
        n_lower_layers  = melt_layers - n_upper_layers 
        lower_layer = range(n_upper_layers, melt_layers)

        primitive_mantle_layer = range(melt_layers, self.z_layers)

        fractionation_factor = 1 - (self.target_SiO2 / self.upper_SiO2)

        #####      DO THE DYANMICS       #############################
        # Set the primitive initial state.  
        for i in primitive_mantle_layer:
            self.grid_cell_state[grid_cell_id][i] = self.primitive_initial_state

        # Weighted average of upper 
        wt_sio2_upper = self.average_target / (1 - fractionation_factor)
            
        # Impact melt portion  (Upper)
        for i in upper_layer:
            self.grid_cell_state[grid_cell_id][i] = wt_sio2_upper
            if self.bound_sio2:
                self.grid_cell_state[grid_cell_id][i] = self.clip_to_sio2_bounds(self.grid_cell_state[grid_cell_id][i])

        # Lower of impact melt portion
        for i in lower_layer:
            numerator = self.average_target - (self.fraction_upper_layer * wt_sio2_upper)
            self.grid_cell_state[grid_cell_id][i] = numerator / self.fraction_lower_layer
            if self.bound_sio2:
                self.grid_cell_state[grid_cell_id][i] = self.clip_to_sio2_bounds(self.grid_cell_state[grid_cell_id][i])

        for i in range(melt_layers):
            self.grid_cell_state[grid_cell_id][i] = np.round(self.grid_cell_state[grid_cell_id][i],1)
    
    #---------------
    def clip_to_sio2_bounds(self, value):
        value = np.max([45, np.min([80, value])])
        return value

    #--------------------------------------------------------------------------------------------------    
    def state_prep(self):
        total_layers = int(self.max_depth_of_impact_melt / self.z_discretized_km)
        for ilon in self.lon_subset:
            for ilat in self.lat_subset:
                grid_cell_id = str(round(ilon,4))+' '+str(round(ilat,4))
                self.grid_cell_state[grid_cell_id] = np.ones(total_layers) * self.primitive_initial_state
        
    #--------------------------------------------------------------------------------------------------    
    def get_average_target(self):
        average_target = 0
        for grid_cell in self.impacted_grid_cells:
            grid_cell_id = str(round(grid_cell[0],4))+' '+str(round(grid_cell[1],4))
            ### If the grid cell has not been hit yet, it is the initial primitive value
            if grid_cell_id in self.grid_cell_state.keys():
                average_target += np.sum(self.grid_cell_state[grid_cell_id][:self.z_layers])
            else:
                average_target += self.primitive_initial_state * self.z_layers
        self.average_target = average_target/(len(self.impacted_grid_cells) * self.z_layers)
        
    #--------------------------------------------------------------------------------------------------    
    def find_the_grid(self, impact_loc):
        self.impacted_grid_cells = [] # first reset the impacted grid cells, then fill them up
        Dmin=10000000
        for ilon in self.lon_subset:
            for ilat in self.lat_subset:
                D = distance(impact_loc[0],ilat,impact_loc[1],ilon)
                if D < Dmin:
                    Dmin = D
                if D <= self.crator_radius:
                    self.impacted_grid_cells.append([ilon, ilat])
        if len(self.impacted_grid_cells) < 1:
            
            # If the crator didn't impact any grids in the subsample, 
            # Check to see if the min distance is smaller than the length of a grid.
            # If it is, we can assign it to the closest grid.
            # If not, then just ignore it.
            if Dmin < 30:
                if self.verbose:
                    print("Warning. There are no grids impacted!")
                    print('Dmin', Dmin, 'crator radius', self.crator_radius, 'impact location', impact_loc)
                for ilon in self.lon_subset:
                    for ilat in self.lat_subset:
                        D = distance(impact_loc[0],ilat,impact_loc[1],ilon)
                        if D == Dmin:
                            self.impacted_grid_cells.append([ilon, ilat])
                            if self.verbose:
                                print('impacting grid cell', [ilon, ilat])

    #--------------------------------------------------------------------------------------------------    
    def loop_impact_grid(self, impactor_diameter):
        for grid_cell in self.impacted_grid_cells:
            grid_cell_id = str(round(grid_cell[0],4))+' '+str(round(grid_cell[1],4))

            ################      DO THE DYANMICS       #############################
            self.state_dynamics(impactor_diameter, grid_cell_id)

            self.test_one_grid_cell(grid_cell_id, impactor_diameter)
    #--------------------------------------------------------------------------------------------------    
    def impact_dimensions(self, impactor_diameter):
            # The impact crator is 10*Diameter, so the radius is half that
            self.crator_diameter = 10*impactor_diameter
            self.crator_radius = self.crator_diameter/2
            # Random between 1-3 to accound for varying impact angle.
            self.impact_angle_factor = random.uniform(self.consider_impact_angle[0], self.consider_impact_angle[1])
            self.penetration_depth = self.impact_angle_factor * impactor_diameter
            self.z_layers = int( np.min([self.max_depth_of_impact_melt, self.penetration_depth]) / self.z_discretized_km )

    #--------------------------------------------------------------------------------------------------    
    def test_one_grid_cell(self, grid_cell_id, impactor_diameter):
        ##### Testing one cell:
        if grid_cell_id == self.impact_test_id:
            self.count_test_hits+=1
            self.test_time.append(self.sim_time)
            self.impactors_at_test_cell.append(impactor_diameter)
            self.average_test_target_list.append(self.average_target)
            self.top_layers_at_test_cell.append(np.mean(self.grid_cell_state[self.impact_test_id][:self.test_layers]))

    #--------------------------------------------------------------------------------------------------
    def re_bin_sio2(self, temp_state, s_min=1, s_max=100, ds=1):
        """
            Functionto place the mean SiO2 into the proper bin for distribution.
        """
        for s in range(s_min,s_max,ds):
            if temp_state<=s:
                return s
            elif temp_state>=s_max:
                return s_max
            else:
                continue

    # ---------------------------------------------------------------------------------------------
    def plot_3x3_map(self, ens1, ens2, ens3,
                    save_figure=False, plot_figure=False,
                    fig_path='./', map_layers=[0], dist_layer=0,
                    bound_plots=True):

        # Columns = ensembles
        ensemble_ids = [ens1, ens2, ens3]

        # Rows = timesteps (time downwards)
        timesteps = [50, 100, 499]

        zscalelow = 44
        zscalehigh = 66

        if not plot_figure and not save_figure:
            return

        # Create 3×3 layout
        fig, axs = plt.subplots(3, 3, figsize=(7, 4.5))
        plt.subplots_adjust(wspace=0.3, hspace=0.25)

        def load_state(ens_id, t_myr):
            """Load a state file for a specific ensemble + timestep."""
            state_file = f"/media/volume/ml_ngen/impaacs/impact_states/july2025/{ens_id}/{t_myr}.pkl"
            with open(state_file, "rb") as fb:
                return pkl.load(fb)

        def build_map():
            """Build 2D SiO2 map array."""
            z = np.zeros([self.n_x, self.n_y])
            for i, ilon in enumerate(self.lon_subset):
                for j, ilat in enumerate(self.lat_subset):
                    grid_cell = f"{round(ilon,4)} {round(ilat,4)}"
                    state_val = np.mean([self.grid_cell_state[grid_cell][i] for i in map_layers])
                    z[i, j] = self.re_bin_sio2(state_val)
            return z

        # === LOOP: rows = times, columns = ensembles ===
        for row, t in enumerate(timesteps):
            for col, ens_id in enumerate(ensemble_ids):

                # Load state for this timestep + ensemble
                impact_states = load_state(ens_id, t)
                self.grid_cell_state = impact_states
                self.sim_time = t * 1_000_000

                # Compute layer distribution (needed for titles, not plotted here)
                self.do_volume_by_layer(n_layers=4)

                # Build 2D map
                Z = build_map()
                X, Y = np.meshgrid(self.lon_subset, self.lat_subset)

                ax = axs[row, col]

                # Plot map
                levels = np.arange(zscalelow, zscalehigh, 2)
                cs = ax.contourf(X, Y, Z.T, levels=levels,
                                cmap=plt.cm.get_cmap("jet", len(levels)))

                # Column headers = ensemble IDs
                if row == 0 and col == 0:
                    ax.set_title(f"Ens {ens_id} (mean)")
                if row == 0 and col == 1:
                    ax.set_title(f"Ens {ens_id} (5th)")
                if row == 0 and col == 2:
                    ax.set_title(f"Ens {ens_id} (95th)")

                # Row labels = time steps
                if col == 0:
                    ax.set_ylabel(f"{t} Myr", fontsize=12)

                ax.set_xlim(self.lon_lims)
                ax.set_ylim(self.lat_lims)
                ax.set_xticks(np.arange(self.lon_lims[0], self.lon_lims[1], 10))
                ax.set_yticks(np.arange(self.lat_lims[0], self.lat_lims[1], 10))

                # Add colorbar only for last column
                if col == len(ensemble_ids) - 1:
                    fig.colorbar(cs, ax=ax)

        # Save or show figure
        if save_figure:
            fig.savefig(f"{fig_path}/combined_sio2_progression.png",
                        bbox_inches='tight', dpi=120)

        if plot_figure:
            plt.show()

        plt.close()

    # ---------------------------------------------------------------------------------------------
    def plot_3x3_distributions(self, ens1, ens2, ens3,
                            save_figure=False, plot_figure=False,
                            fig_path='./', dist_layer=0,
                            map_layers=[0]):

        # Columns = ensembles
        ensemble_ids = [ens1, ens2, ens3]

        # Rows = timesteps (time downwards)
        timesteps = [50, 100, 499]

        if not plot_figure and not save_figure:
            return

        # Smaller figure, matching new map layout (7×4 inches)
        fig, axs = plt.subplots(3, 3, figsize=(7, 4.5))
        plt.subplots_adjust(wspace=0.3, hspace=0.25)

        def load_state(ens_id, t_myr):
            state_file = f"/media/volume/ml_ngen/impaacs/impact_states/july2025/{ens_id}/{t_myr}.pkl"
            with open(state_file, "rb") as fb:
                return pkl.load(fb)

        # === LOOP: rows = times, columns = ensembles ===
        for row, t in enumerate(timesteps):
            for col, ens_id in enumerate(ensemble_ids):

                # Load state for this timestep + ensemble
                impact_states = load_state(ens_id, t)
                self.grid_cell_state = impact_states
                self.sim_time = t * 1_000_000

                # Compute distribution
                self.do_volume_by_layer(n_layers=4)

                ax = axs[row, col]

                # Extract distribution for selected layer
                keys = list(self.percent_volume_by_layer[dist_layer].keys())   # SiO2 bins
                vals = list(self.percent_volume_by_layer[dist_layer].values()) # Percent volumes

                ax.bar(keys, vals, width=1.2)

                # Column titles = ensemble quantile meaning
                if row == 0:
                    if col == 0:
                        ax.set_title(f"Ens {ens_id} (mean)")
                    elif col == 1:
                        ax.set_title(f"Ens {ens_id} (5th)")
                    elif col == 2:
                        ax.set_title(f"Ens {ens_id} (95th)")

                # Row label = timestep
                if col == 0:
                    ax.set_ylabel(f"{t} Myr", fontsize=12)

                # Axis formatting
                ax.set_xlabel("Surface SiO₂ content")
                ax.set_ylim([0, 50])
                ax.set_xlim([44, 66])
                ax.set_xticks(range(44, 67, 5))

        # Save/show output
        if save_figure:
            fig.savefig(f"{fig_path}/combined_sio2_distributions.png",
                        bbox_inches='tight', dpi=120)

        if plot_figure:
            plt.show()

        plt.close()

    # ---------------------------------------------------------------------------------------------
    def plot_map_and_bar(self, save_figure=False, plot_figure=False, fig_path='./', map_layers=[0], dist_layer=0):
        
        """
            Function for plotting 2D map of SiO2 States.
            Function inputs:
                save_figure=False
                plot_figure=False
                fig_path='./'
        """
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titlepad": 4,
            "axes.labelpad": 2,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2
        })
        if not plot_figure and not save_figure:
            print('not plotting figure')
            return
        print(f"plotting SiO2 map for layers {[i for i in map_layers]}, and distribution for layer {dist_layer}")
        z = np.zeros([self.n_x, self.n_y])
        for i, ilon in enumerate(self.lon_subset):
            for j, ilat in enumerate(self.lat_subset):
                grid_cell = str(round(ilon,4))+' '+str(round(ilat,4))
                temp_state = np.mean([self.grid_cell_state[grid_cell][i] for i in map_layers])
                temp_state = self.re_bin_sio2(temp_state)
                z[i, j] = temp_state
        X, Y = np.meshgrid(self.lon_subset, self.lat_subset)
        fig = plt.figure(figsize=(6.5, 3))
        plt.subplots_adjust(left=0.001, right=0.999, bottom=0.01, top=0.99)
        grid = plt.GridSpec(1, 30, wspace=0.01, hspace=0.01)
        plt.subplot(grid[0, :22])
        cmap = cm.jet
        levels = np.arange(44, 64, 2)
        cs = plt.contourf(X, Y, np.transpose(z), levels, cmap=cm.get_cmap(cmap, len(levels) - 1)) 
        cbar = fig.colorbar(cs, pad=0.01)
        plt.title(f'Surface SiO2 % at {int(self.sim_time/1000000)}myr')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        ax = plt.subplot(grid[0, 22:])
        ax.bar(
            list(self.percent_volume_by_layer[dist_layer].keys()),
            list(self.percent_volume_by_layer[dist_layer].values()),
            width=1.2
        )
        ax.set_xlabel('Surface SiO2 %')
        ax.set_title('PDF')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: v / 100))
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        if save_figure:
            plt.savefig(
                fig_path + '{}myr.png'.format(int(self.sim_time/1000000)),
                bbox_inches='tight',
                dpi=100
            )
        if plot_figure:
            plt.show()
        plt.close()


    # ---------------------------------------------------------------------------------------------
    def calculate_relative_percent_crust_vol_multiplier(self):
        # Make the calculations to plot the raltive percent crustal volume
        # (grid_size [m] / 1000 [km/m])^2 [km^2] * Impc.z_discretized_km [km]
        
        grid_area = np.power(self.egrid.map_scale/1000,2)
        
        cube_volume = grid_area * self.z_discretized_km
        print(f"cube volume {cube_volume}, in km^3")
        
        # Get our surface are in proportion to Earth surface area
        # grid area [km^2] * n_grids in row * n_grid in col
        n_cubes_per_layer = self.n_x * self.n_y
        sample_area = grid_area * n_cubes_per_layer
        print(f"sample area {sample_area}, in km^2")
        surface_area_of_earth = 507637669.626 #km^2
        sample_area_ratio = sample_area / surface_area_of_earth
        print(f"our sample represents {np.round(sample_area_ratio,3)} of earth's surface area")
        surface_area_multiplier = 1/sample_area_ratio
        print(f"we need to multiply our volume by {np.round(surface_area_multiplier,3)} to correct for sample/earth area")
        cube_volume_multiplier = cube_volume * surface_area_multiplier
        print(f"multiply n_cubes by {np.round(cube_volume_multiplier,1)} to get crust volume [km^3] on earth")
        total_current_crust = 7.2e9 #km^3
        print(f"divide by {total_current_crust} to get relative percent crust volume")
        self.relative_percent_crust_vol_multiplier = cube_volume_multiplier / total_current_crust
        print(f"the final multiplier to get relative percent volume crust is {self.relative_percent_crust_vol_multiplier}")

    # ---------------------------------------------------------------------------------------------
    def do_volume_by_layer(self, n_layers=10, sio2_threshold=58):
        
        """
            Function Summarizing and saving total number of grids at specific SiO2 (mean) in a sample region.
                                 and saving SiO2 percentages in a sample region.
            Function inputs:
                plot_x_lims = Limits of longitude for SiO2 sample
                plot_y_lims = Limits of latitude for SiO2 sample
                n_layers = number of discretized layers to include in the average.
                sio2_threshold (int): The threshold at which to count the cells as crust, default = 58
            Function outputs:
                sum_at_sio2_by_layer = total number of cells with greater than threshold
                percent_volume_by_layer = percentage of cells (per layer) greater than threshold
        """

        self.percent_volume_by_layer={} 
        self.sum_at_sio2_by_layer={} 

        for i_layer in range(n_layers):
            z = np.zeros([self.n_x, self.n_y])
            bar_list = []
            for i, ilon in enumerate(self.lon_subset):
                for j, ilat in enumerate(self.lat_subset):
                    grid_cell = str(round(ilon,4))+' '+str(round(ilat,4))
                    temp_state = self.grid_cell_state[grid_cell][i_layer]
                    temp_state = self.re_bin_sio2(temp_state)
                    z[i, j] = temp_state
                    
                    mean_sio2 = self.grid_cell_state[grid_cell][i_layer]
                    
                    if not np.isnan(mean_sio2):
                        bar_list.append(self.re_bin_sio2(mean_sio2))
            
            bar_list = [x for x in bar_list if x != None]
            
            percent_data = {}
            n_cells = {}
            for u in np.unique(bar_list):
                n_cells[u] = bar_list.count(u)
                percent_data[u] = 100*bar_list.count(u)/len(bar_list)
    
            self.percent_volume_by_layer[i_layer] = percent_data
            self.sum_at_sio2_by_layer[i_layer] = n_cells
       
        n_cubes_above_threshold = 0
        for i_layer in range(n_layers):
            for sio2_bin in list(self.sum_at_sio2_by_layer[i_layer].keys()):
                if sio2_bin >= self.sio2_threshold:
                    n_cubes_above_threshold += self.sum_at_sio2_by_layer[i_layer][sio2_bin]
        self.n_cubes_above_threshold_list.append(n_cubes_above_threshold)
        rel_vol_perc_crust_at_time = n_cubes_above_threshold * self.relative_percent_crust_vol_multiplier
        self.relative_percent_crust_vol_list.append(rel_vol_perc_crust_at_time)

 
    # ---------------------------------------------------------------------------------------------
    def get_subset_of_grids(self):
        for ilon in self.egrid.londim:
            for ilat in self.egrid.latdim:
                if ilat > self.lat_lims[0] and ilat < self.lat_lims[1]:
                    if ilon > self.lon_lims[0] and ilon < self.lon_lims[1]:
                        if ilat not in self.lat_subset:
                            self.lat_subset.append(ilat)
                        if ilon not in self.lon_subset:
                            self.lon_subset.append(ilon)


