#!/home/jmframe/programs/anaconda3/bin/python3
import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from matplotlib import cm
import random
from math import sin, cos, sqrt, atan2, radians
from ease_grid import EASE2_grid
grid_size = 36000
egrid = EASE2_grid(grid_size)
import time
import impacts

################
################
################
SUB_FOLDER_NAME="july2025"


#-----------------------------------------------------------------------------------
# Loop through the ensemble members. Want to calculate the probabilities at each go.
for ensemble_member in range(1,5):

    # Set the size bins
    max_diameter=330
    diam_bins = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500]
    diam_labs = [f'{str(i).zfill(3)}-{str(j).zfill(3)}' for i, j in zip(diam_bins[0:-1], diam_bins[1:])]
    diam_range = {f'{str(i).zfill(3)}-{str(j).zfill(3)}':[i,j] for i, j in zip(diam_bins[0:-1], diam_bins[1:])}
    lambda_start = {j:1+i/10 for i, j in enumerate(list(diam_range.keys()))}
    lambda_end = {j:1+(np.power(i,1.03)) for i, j in enumerate(list(diam_range.keys()))}
    
    percent_dict={}
    
    # Set some time steps to save the full state
    list_impacts_export = list(range(0,500,25))
    list_impacts_export.append(499)

    # Associate seed with ensemble member, so we can compare different scenarios
    random.seed(ensemble_member)
    np.random.seed(ensemble_member)

    # Make dictionary with size bins and frequency
    with open('sfd.csv', 'r') as f:
        freqs = pd.read_csv(f).groupby('D').sum()
    los_dict = {i:0 for i in diam_labs}
    his_dict = {i:0 for i in diam_labs}
    for i in freqs.index.values:
        for j in range(len(diam_labs)):
            if i < diam_bins[j+1] and i >= 5:
                los_dict[diam_labs[j]] += freqs.loc[i,'low']
                his_dict[diam_labs[j]] += freqs.loc[i,'high']
                break
            elif i >= diam_bins[-1]:
                los_dict[diam_labs[-1]] += freqs.loc[i,'low']
                his_dict[diam_labs[-1]] += freqs.loc[i,'high']
                break
    df_freq = pd.DataFrame.from_dict({'high':his_dict, 'low':los_dict, 
                                      'lambda_start':lambda_start, 'lambda_end':lambda_end})
    
    df_freq['frequency_factor'] = [.1+i/10 for i in range(len(diam_bins)-1)]

    print("impact frequency")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_freq)
    
    t_total=500
    
    fivehundredmillion = 500000000
    freq_factor = fivehundredmillion/t_total
    
    not_converged=True
    while not_converged:
    
        pp = {x:np.zeros(t_total) for x in diam_labs}
        l = {x:np.linspace(y,z,t_total) for x,y,z in zip(diam_labs,df_freq['lambda_start'],df_freq['lambda_end'])}
    
        for D in diam_labs:
            pp[D] = l[D]*np.exp(-l[D])
    
        df = pd.DataFrame(data=pp)
        hits = {d:np.zeros(t_total) for d in diam_labs}
    
        # Main loop through time. Calculate the total number of impacts of each diameter at each time step
        for t in range(0,t_total):
            for D in diam_labs:
                hits[D][t] = np.floor(np.random.poisson(pp[D][t] / df_freq.loc[D, 'frequency_factor']))
    
    
        total_sum = np.sum([hits[d] for d in diam_labs])
        sums = {d:np.sum(hits[d]) for d in diam_labs}
        frac = {d:np.round(np.sum(hits[d])/total_sum,2) for d in diam_labs}
        
        for d in diam_labs:
            df_freq.loc[d,'total']=sums[d]
        
        good_numbers = 0
        for d in diam_labs:
            if df_freq.loc[d,'total'] < df_freq.loc[d,'low']:
                df_freq.loc[d,'frequency_factor'] = df_freq.loc[d,'frequency_factor']*random.random()
            elif df_freq.loc[d,'total'] > df_freq.loc[d,'high']:
                df_freq.loc[d,'frequency_factor'] = df_freq.loc[d,'frequency_factor']*(1+random.random())
            else:
                good_numbers+=1
        if good_numbers == df_freq.shape[0]:
            not_converged = False
    print("impact frequency")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_freq)
        df_freq.to_csv("impact_probabilities_export_{}/ensemble_{}.csv".format(SUB_FOLDER_NAME, ensemble_member))

    impact_time = np.linspace(0,fivehundredmillion,t_total+1)[1:]
    df = pd.DataFrame(data=hits, index=impact_time)

    print("Running ensemble {}".format(ensemble_member))
    impact_boundz=20
    [-impact_boundz, impact_boundz]

    #=======================================================================#
    #=======================================================================#
    Impc = impacts.IMPAaCS(egrid,                                           #
                           max_depth_of_impact_melt=998,                    #
                           ensemble=ensemble_member,                        #
                           verbose=False,                                   #
                           lon_lims = [-impact_boundz, impact_boundz],      #
                           lat_lims = [-impact_boundz, impact_boundz],      #
                           bound_sio2=True,                                 #
                           z_discretized_km=int(1))                         #
    #=======================================================================#
    #=======================================================================#

    diam_labs.reverse()

    for it, t in enumerate(df.index.values):
        start_time = time.time()
        for d in diam_labs:
            for i in range(int(df.loc[t,d])):
    
                # locate the the impacts on earth
                impact_lat = random.randrange(-90,90)
                impact_lon = random.randrange(-180,180)
                if np.abs(impact_lat) > impact_boundz:
                    continue
                if np.abs(impact_lon) > impact_boundz:
                    continue
                impact_loc = [impact_lat, impact_lon]
                impactor_diameter = random.randrange(diam_range[d][0],diam_range[d][1])
                    
                #####      DO THE DYANMICS       #############################
                Impc.update(impact_loc, impactor_diameter, t)
                
        n_layers_for_percent_volume = 12
        Impc.do_volume_by_layer(n_layers = n_layers_for_percent_volume)
    
        # At every time step save the total relative crust over time
        with open(f'sio2_percent_tables/{SUB_FOLDER_NAME}/total_relative_crust_{ensemble_member}.txt', 'w') as f:
            for item in Impc.relative_percent_crust_vol_list:
                f.write("%s\n" % item)
       
        for i_layer in range(n_layers_for_percent_volume):
            if it == 0:
                percent_dict[i_layer] = pd.DataFrame(
                    Impc.percent_volume_by_layer[i_layer], index=[it]
                )
            else:
                new_row = pd.DataFrame(
                    Impc.percent_volume_by_layer[i_layer], index=[it]
                )
                percent_dict[i_layer] = pd.concat(
                    [percent_dict[i_layer], new_row],
                    ignore_index=True
                )

        if it in list_impacts_export:
            print('time', it)
            Impc.plot_map_and_bar(save_figure=False,plot_figure=False,fig_path="./figs/ensemble_figs/{}/".format(ensemble_member))
            print("elapsed time: {}".format(time.time() - start_time))
            print(percent_dict[0].iloc[-1,:])
            print(Impc.test_time)
            print(Impc.average_test_target_list)
            print(Impc.top_layers_at_test_cell)
            with open('impact_states/{}/{}/{}.pkl'.format(SUB_FOLDER_NAME, ensemble_member, it), 'wb') as fb:
                pkl.dump(Impc.grid_cell_state, fb, pkl.HIGHEST_PROTOCOL)
    
    for i_layer in range(12):
        percent_dict[i_layer].to_csv("sio2_percent_tables/{}/ensemble_{}_{}.csv".format(SUB_FOLDER_NAME, ensemble_member, i_layer))
