#!/home/jmframe/programs/anaconda3/bin/python3

import numpy as np
import pandas as pd
import random
from math import sin, cos, sqrt, atan2, radians
from ease_grid import EASE2_grid
from matplotlib import pyplot as plt

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
    
    count_test_hits = 0
    impact_test_id = str(round(-87.5726,4))+' '+str(round(33.2921,4))
    grid_cell_state = {}
    impacted_grid_cells = []
    impactors_at_test_cell = [0]
    test_time = [0]
    
    def __init__(self, egrid, 
                 max_depth_of_impact_melt=330,
                 ensemble = 0,
                 primitive_initial_state=45,
                 fraction_upper_layer = 2/3,
                 target_SiO2 = 62.58,     # From sudbury
                 upper_SiO2 = 68.71,      # From sudbury
                 n_layers_impact_melt = 2,
                 z_discretized_km = int(2),
                 proportion_melt_from_impact = 1/3,
                 sim_time=0):
        self.egrid = egrid
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
        self.top_layer_at_test_cell = [self.primitive_initial_state]
        self.average_test_target_list = [self.primitive_initial_state]
        self.n_x = self.egrid.londim.shape[0]
        self.n_y = self.egrid.latdim.shape[0]
        self.sim_time=sim_time
        self.state_prep()

    #--------------------------------------------------------------------------------------------------
    def update(self, impact_loc, impactor_diameter, sim_time=0):
        self.sim_time = sim_time
        self.impact_dimensions(impactor_diameter)
        self.find_the_grid(impact_loc)
        self.get_average_target(impactor_diameter)
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
        depth_of_impact_melt = impactor_diameter * self.proportion_melt_from_impact # D/3

        #Vertical discretization.
        melt_layers = int(depth_of_impact_melt / self.z_discretized_km)

        lower_layer  = range(int(round(self.fraction_upper_layer * melt_layers,2)), melt_layers)
        upper_layer  = range(0, int(round(self.fraction_upper_layer * melt_layers,2)))

        fracionated_melt = depth_of_impact_melt * self.fraction_upper_layer #Units: km

        fractionation_factor = 1 - (self.target_SiO2 / self.upper_SiO2)

        #####      DO THE DYANMICS       #############################
        # Set lower layer to primitive initial state.  
        for i in lower_layer:
            self.grid_cell_state[grid_cell_id][i] = self.primitive_initial_state

        # Impact melt portion  (Upper)
        for i in upper_layer:
            self.grid_cell_state[grid_cell_id][i] = self.average_target / (1 - fractionation_factor)

        # Weighted average of upper    
        wt_sio2_upper = self.grid_cell_state[grid_cell_id][0]

        # Lower of impact melt portion
        for i in lower_layer:
            numerator = self.average_target-(self.fraction_upper_layer * wt_sio2_upper)
            self.grid_cell_state[grid_cell_id][i] = numerator / self.fraction_lower_layer

        for i in range(melt_layers):
            self.grid_cell_state[grid_cell_id][i] = np.round(self.grid_cell_state[grid_cell_id][i],1)
    
    #--------------------------------------------------------------------------------------------------    
    def state_prep(self):
        total_layers = int(self.max_depth_of_impact_melt / self.z_discretized_km)
        for ilon in self.egrid.londim:
            for ilat in self.egrid.latdim:
                grid_cell_id = str(round(ilon,4))+' '+str(round(ilat,4))
                self.grid_cell_state[grid_cell_id] = np.ones(total_layers) * self.primitive_initial_state
    
    #--------------------------------------------------------------------------------------------------    
    def get_average_target(self, impactor_diameter):
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
        for ilon in self.egrid.londim:
            for ilat in self.egrid.latdim:
                D = distance(impact_loc[0],ilat,impact_loc[1],ilon)
                if D < Dmin:
                    Dmin = D
                if D <= self.crator_radius:
                    self.impacted_grid_cells.append([ilon, ilat])
        if len(self.impacted_grid_cells) < 1:
            print("Warning. There are no grids impacted!")
            print('Dmin', Dmin, 'crator radius', self.crator_radius, 'impact location', impact_loc)
            for ilon in self.egrid.londim:
                for ilat in self.egrid.latdim:
                    D = distance(impact_loc[0],ilat,impact_loc[1],ilon)
                    if D == Dmin:
                        self.impacted_grid_cells.append([ilon, ilat])
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
            self.z_layers = int(np.ceil(impactor_diameter / self.z_discretized_km))

    #--------------------------------------------------------------------------------------------------    
    def test_one_grid_cell(self, grid_cell_id, impactor_diameter):
        ##### Testing one cell:
        if grid_cell_id == self.impact_test_id:
            self.count_test_hits+=1
            self.test_time.append(self.sim_time)
            self.impactors_at_test_cell.append(impactor_diameter)
            self.average_test_target_list.append(self.average_target)
            self.top_layer_at_test_cell.append(self.grid_cell_state[self.impact_test_id][0])
    
    #--------------------------------------------------------------------------------------------------
    def plot_map(self):
        z = np.zeros([self.n_x, self.n_y])
        for i, ilon in enumerate(self.egrid.londim):
            for j, ilat in enumerate(self.egrid.latdim):
                grid_cell = str(round(ilon,4))+' '+str(round(ilat,4))
                z[i, j] = np.mean(self.grid_cell_state[grid_cell][0:2])
        X, Y = np.meshgrid(self.egrid.londim, self.egrid.latdim)
        plt.contourf(X, Y, np.transpose(z))
        plt.colorbar()
        plt.title('top 6-meter sio2 at time {}myr'.format(int(self.sim_time/1000000)))
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.savefig('./figs/maps/E{}_sio2_map_at_time_{}myr.png'.format(self.ensemble, int(self.sim_time/1000000)), 
                    bbox_inches='tight')
        plt.close()
