#!/home/jmframe/programs/anaconda3/bin/python3

import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from ease_grid import EASE2_grid
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import cm
import random

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
                 consider_impact_angle=False):
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
        
        self.sum_at_sio2_by_layer = {}
        self.percent_volume_by_layer = {}
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
        value = np.max([40, np.min([80, value])])
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
            if self.consider_impact_angle:
                self.impact_angle_factor = random.uniform(1, 3)
            else:
                self.impact_angle_factor = 1
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
    def plot_map_and_bar(self, save_figure=False, plot_figure=False, fig_path='./', map_layers=[0], dist_layer=0, bound_plots=True):
        
        """
            Function for plotting 2D map of SiO2 States.
            Function inputs:
                save_figure=False
                plot_figure=False
                fig_path='./'
        """
        
        if not plot_figure and not save_figure:
            print('not plotting figure')
            return

        print(f"plotting SiO2  map for layers {[i for i in map_layers]}, and distribution for layer {dist_layer}")

        z = np.zeros([self.n_x, self.n_y])
        bar_list = []
        for i, ilon in enumerate(self.lon_subset):
            for j, ilat in enumerate(self.lat_subset):
                grid_cell = str(round(ilon,4))+' '+str(round(ilat,4))
                temp_state = np.mean([self.grid_cell_state[grid_cell][i] for i in map_layers])
                temp_state = self.re_bin_sio2(temp_state)
                z[i, j] = temp_state
        
        X, Y = np.meshgrid(self.lon_subset, self.lat_subset)
        
        fig = plt.figure(figsize=(12, 7))

        grid = plt.GridSpec(1, 7, wspace = .1, hspace = .1)
        plt.subplots_adjust(wspace= 0.1, hspace= 0.1)

        plt.subplot(grid[0, :5])
        
        if bound_plots:
            levels = np.arange(36, 86, 2)
        cmap = cm.jet
        cs = plt.contourf(X, Y, np.transpose(z), levels, cmap=cm.get_cmap(cmap, len(levels) - 1)) 
        if bound_plots:
            cbar = fig.colorbar(cs, ticks=range(36,86,2))
        else:
            cbar = fig.colorbar(cs)

        plt.title('Surface SiO2 content at {}myr, layers {}'.format(int(self.sim_time/1000000), map_layers))
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        if bound_plots:
            plt.xlim(self.lon_lims)
            plt.ylim(self.lat_lims)
            plt.xticks(np.arange(self.lon_lims[0], self.lat_lims[1], 10))
        
        plt.subplot(grid[0, 5:])
        plt.bar(list(self.percent_volume_by_layer[dist_layer].keys()), list(self.percent_volume_by_layer[dist_layer].values()), width=1.2)
        if bound_plots:
            plt.xlim([35,85])
            plt.ylim([0,50])
            plt.xticks(np.arange(35, 85, 5))
        plt.xlabel('Surface SiO2 content')
        plt.ylabel(f'Percent volume for layer {dist_layer}')
        if save_figure:
            plt.savefig(fig_path+'{}myr.png'.format(int(self.sim_time/1000000)), 
                    bbox_inches='tight', dpi = 100)
        if plot_figure:
            plt.show()
        plt.close()
        
    # ---------------------------------------------------------------------------------------------
    def do_volume_by_layer(self, n_layers=1):
        
        """
            Function Summarizing and saving total number of grids at specific SiO2 (mean) in a sample region.
                                 and saving SiO2 percentages in a sample region.
            Function inputs:
                plot_x_lims = Limits of longitude for SiO2 sample
                plot_y_lims = Limits of latitude for SiO2 sample
                n_layers = number of discretized layers to include in the average.
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


