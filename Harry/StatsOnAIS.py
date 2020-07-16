#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:33:33 2020

@author: harry newton
"""


import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
from haversine import haversine, Unit


#%%
shipTypes=[
"drifting_longlines", # for debugging picking just one
"fixed_gear",
"pole_and_line",
"purse_seines",
"trawlers",
"trollers",
"unknown"
]
columnNames=['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port',
       'speed', 'course', 'lat', 'lon', 'is_fishing', 'source']
dataTypes={'mmsi':float,'timestamp':float,'distance_from_shore':float, \
            'distance_from_port':float, 'speed':float, 'course':float, \
            'lat':float,'lon':float, 'is_fishing':object, 'source':object}
dataUnits={'mmsi':'VesselID','timestamp':'seconds','distance_from_shore':'meters', \
            'distance_from_port':'meters', 'speed':'knots', 'course':'degrees', \
            'lat':'decimal degrees','lon':'decimal degrees', 'is_fishing':'label', 'source':'text'}
bigDF=pd.DataFrame(columns=columnNames) #,dtype=dataTypes)

#%%
#Read csv file for each shipType into bigDF
path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_data"
#data available at: https://globalfishingwatch.org/data-download/datasets/public-training-data-v1

for s in shipTypes:
        df = pd.read_csv(os.path.join(path,s+".csv"),dtype=dataTypes)
        print("S: ", s, "Records: ",len(df)); 
        df['type']=s
     #   df['mmsi'] = df['mmsi'].astype(int)
     #   print("Features: ",df.columns)
        bigDF=bigDF.append(df,ignore_index=True)
        
#%% 
# to determine long voyage risk factor, need stats on voyages
normsCols=['numObs','numShips','meanVoyage','sdVoyage','numVoyage']
norms=pd.DataFrame(columns=normsCols,dtype=object)

#Determine number of observations (signals) and unique ships for each ship type 
norms.numObs=bigDF.groupby("type")["mmsi"].count()

norms.numShips=bigDF.groupby("type")["mmsi"].nunique()

#%%
# Prepare for risk factors on gaps in ship signals (both time & distance)

#Make sure observations are ordered first by ship then time observed
bigDF=bigDF.sort_values(['mmsi','timestamp'])

# Determine the time interval
bigDF['ztime_diff'] = bigDF.timestamp - bigDF.timestamp.shift(1)
print('Num NaN:',bigDF.ztime_diff.isnull().sum())

# Eliminate meaningless intervals when the prev row was for a different ship
condition=bigDF.mmsi!=bigDF.mmsi.shift(1)
bigDF.loc[condition,'ztime_diff']=np.nan
print('After ship changes, Num NaN:',bigDF.ztime_diff.isnull().sum())
#for test dataset of trawlers, should be 49

print(bigDF.ztime_diff.describe())

for s in shipTypes:
    condition=bigDF.type==s
    sDF=bigDF.loc[condition,'ztime_diff']
    norms.at[s,'time_diff_sd']=sDF.std()
    norms.at[s,'time_diff_q75']=sDF.quantile(q=0.75)
    norms.at[s,'time_diff_mean']=sDF.mean()

#Plan to use the q=.75 but will run by SMEs. 
#%%    
# Determine distance between signals for each voyage of each ship
#For now, use haverstine on each pair of observations for dist travelled
#https://pypi.org/project/haversine/ 
#Later, could optimize using S2 library or
#distance_to_shore as a surrogate if this proves
#too costly since that was already computed in the dataset

#Will assign voyage unique IDs while computing distances
#with logic that within 500 meters of shore  is not a voyage   

bigDF['dist_diff']=np.nan
bigDF['voyID']=np.nan
bigDF['aveSpeed']=np.nan
col4voyID=bigDF.columns.get_loc('voyID')
col4mmsi=bigDF.columns.get_loc('mmsi')
col4dist=bigDF.columns.get_loc('distance_from_shore')
col4diff=bigDF.columns.get_loc('dist_diff')
col4lat=bigDF.columns.get_loc('lat')
col4lon=bigDF.columns.get_loc('lon')
col4aveSpeed=bigDF.columns.get_loc('aveSpeed')
col4ztime_diff=bigDF.columns.get_loc('ztime_diff')

voyID=1  #each voyage will get a unique ID
bigDF.iat[0,col4voyID]=voyID
flag_atSea=False
for i in range(1,len(bigDF)):
    #if at sea & same ship, compute distance and assign voyageID
    if bigDF.iat[i,col4dist]>500: #at sea
        flag_atSea=True
        if bigDF.iat[i,col4mmsi] == bigDF.iat[i-1,col4mmsi]: #same ship
            bigDF.iat[i,col4voyID]=voyID
            bigDF.iat[i,col4diff] = \
                haversine((bigDF.iat[i,col4lat],bigDF.iat[i,col4lon]), \
                          (bigDF.iat[i-1,col4lat],bigDF.iat[i-1,col4lon]))
            if (bigDF.iat[i,col4ztime_diff]>=60): #>= 1 min
                bigDF.iat[i,col4aveSpeed]=bigDF.iat[i,col4diff] \
                                            /bigDF.iat[i,col4ztime_diff]
            else:
                bigDF.iat[i,col4aveSpeed]=NaN
        else:
            voyID=voyID+1
    else:               #not at sea              
        if flag_atSea:  #only increment voyID if was last at sea
            voyID=voyID+1
        flag_atSea=False
        

#this code no longer needed...            
# Eliminate meaningless intervals when the prev row was for a different ship
# condition=bigDF.mmsi!=bigDF.mmsi.shift(1)
# bigDF.loc[condition,'dist_diff']=np.nan

print('After ship changes, Num NaN:',bigDF.dist_diff.isnull().sum())
print(bigDF.dist_diff.describe())

#store statistics
for s in shipTypes:
    condition=bigDF.type==s
    sDF=bigDF.loc[condition,'dist_diff']
    norms.at[s,'dist_diff_sd']=sDF.std()
    norms.at[s,'dist_diff_q']=sDF.quantile(q=0.75)
    norms.at[s,'dist_diff_mean']=sDF.mean()

#%%

#Establish longest voyage for each ship
shipStats=pd.DataFrame()
voyageStats=pd.DataFrame()

voyageStats['len_seconds']=bigDF.groupby('voyID').timestamp.max() \
                   -bigDF.groupby('voyID').timestamp.min()
voyageStats['mmsi']=bigDF.groupby('voyID').mmsi.min()
voyageStats['shipType']=bigDF.groupby('voyID').type.min()
voyageStats['max_dist_gap']=bigDF.groupby('voyID').dist_diff.max()
voyageStats['max_time_gap']=bigDF.groupby('voyID').ztime_diff.max()
voyageStats['max_speed']=bigDF.groupby('voyID').speed.max()
voyageStats['max_aveSpeed']=bigDF.groupby('voyID').aveSpeed.max()

shipStats['shipType']=voyageStats.groupby('mmsi').type.min()                   
shipStats['max_len_voy']=voyageStats.groupby('mmsi').len_seconds.max()
shipStats['max_time_gap']=voyageStats.groupby('mmsi').max_time_gap.max()
shipStats['max_dist_gap']=voyageStats.groupby('mmsi').max_dist_gap.max()
shipStats['max_speed']=voyageStats.groupby('mmsi').speed.max()
shipStats['max_aveSpeed']=voyageStats.groupby('mmsi').aveSpeed.max()

print(norms.describe(include='all'))
print(shipStats.describe(include='all'))
print(voyageStats.describe(include='all'))

#%%
# Export Results
path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_results"

norms.to_csv(os.path.join(path,'norms.csv'))
shipStats.to_csv(os.path.join(path,'shipstats.csv'))
voyageStats.to_csv(os.path.join(path,'voyageStats.csv'))
