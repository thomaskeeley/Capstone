#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: 
 1. Work through millions of AIS signals from ships in fishing industry
    - Create a baseline of normal behavior based on their fishing type
 2. Aggregate each ships signals into voyages (times at sea, not port)
 3. Determine each ships behaivor over its voyages compared to baseline
 4. Write out results for other functions to visualize and determine 
    which ships have operations that may indicate use of Human Trafficking 
Created on Thu Jun 25 11:33:33 2020

Data: 
    Dataset of 26 Million AIS signals over a 5 year period for a selected
    group of 350 ships.   
    
    URL: https://globalfishingwatch.org/data-download/datasets/public-training-data-v1

    Provider: Global Fishing Watch based on a partnership explained in the paper
        
    Terms of Use: https://globalfishingwatch.org/datasets-and-code/

@author: harry newton
"""

import os
#from pathlib import Path
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from datetime import datetime, timedelta
#import itertools
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


for s in shipTypes:
        df = pd.read_csv(os.path.join(path,s+".csv"),dtype=dataTypes)
        print("S: ", s, "Records: ",len(df)); 
        df['type']=s
     #   df['mmsi'] = df['mmsi'].astype(int)
     #   print("Features: ",df.columns)
        bigDF=bigDF.append(df,ignore_index=True)
        
#%% 
# to determine long voyage risk factor, need stats on voyages
normsCols=['numObs','numShips','numVoyages','aveNumVoyages','numVoyages_q25']
norms=pd.DataFrame(columns=normsCols,dtype=object)

#Determine number of observations (signals) and unique ships for each ship type 
norms.numObs=bigDF.groupby("type")["mmsi"].count()

norms.numShips=bigDF.groupby("type")["mmsi"].nunique()

#%%
# Prepare for risk factors on gaps in ship signals (both time & distance)
# Note that at this point, obs are not separated into voyages
#Make sure observations are ordered first by ship then time observed
#bigDF=bigDF.sort_values(['mmsi','timestamp'])

# Determine the time interval
# bigDF['time_diff'] = bigDF.timestamp - bigDF.timestamp.shift(1)
# print('Num NaN:',bigDF.time_diff.isnull().sum())

# Eliminate meaningless intervals when the prev row was for a different ship
# condition=bigDF.mmsi!=bigDF.mmsi.shift(1)
# bigDF.loc[condition,'time_diff']=np.nan
# bigDF.loc[condition,'dist_diff']=np.nan
print('After data load...')

#for test dataset of trawlers, should be 49
print(bigDF.type.describe())

# for s in shipTypes:
#     condition=bigDF.type==s
#     sDF=bigDF.loc[condition,'time_diff']
#     norms.at[s,'time_diff_sd']=sDF.std()
#     norms.at[s,'time_diff_q75']=sDF.quantile(q=0.75)
#     norms.at[s,'time_diff_mean']=sDF.mean()

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

#setup for summary stats between each observation    
bigDF['time_diff']=np.nan
bigDF['dist_diff']=np.nan
#bigDF['aveSpeed']=np.nan  #for later functionality
col4time_diff=bigDF.columns.get_loc('time_diff')
col4dist_diff=bigDF.columns.get_loc('dist_diff')
#col4aveSpeed=bigDF.columns.get_loc('aveSpeed')

#reference columns to compute time and distance deltas
col4lat=bigDF.columns.get_loc('lat')
col4lon=bigDF.columns.get_loc('lon')
col4timestamp=bigDF.columns.get_loc('timestamp')
col4dist2shore=bigDF.columns.get_loc('distance_from_shore')

#setup to label each voyage
bigDF['voyID']=1
col4voyID=bigDF.columns.get_loc('voyID')
col4mmsi=bigDF.columns.get_loc('mmsi')

#Assign first row since it can't be computed with delta formulas
voyID=1  #each voyage will get a unique ID
flag_used_voyID=False

i=0
if bigDF.iat[i,col4dist2shore]>500: #at sea
        flag_atSea=True
        bigDF.iat[i,col4voyID]=voyID
        flag_used_voyID=True

#bigDF.iat[0,col4time_diff] already NaN
#bigDF.iat[0,col4dist_diff] already NaN

#Starting index at 1 because the 0th row has no i-1 row for deltas.
flag_atSea=False
for i in range(1,len(bigDF)):
    #if at sea & same ship, compute distance and assign voyageID
    if bigDF.iat[i,col4dist2shore]>500: #at sea
        flag_atSea=True
        if bigDF.iat[i,col4mmsi] == bigDF.iat[i-1,col4mmsi]: #same ship
            bigDF.iat[i,col4voyID]=voyID
            flag_used_voyID=True
            timeDelta=bigDF.iat[i,col4timestamp]-bigDF.iat[i-1,col4timestamp]
            bigDF.iat[i,col4time_diff]=timeDelta
            distDelta=haversine((bigDF.iat[i,col4lat],bigDF.iat[i,col4lon]), \
                          (bigDF.iat[i-1,col4lat],bigDF.iat[i-1,col4lon]),\
                           unit=Unit.NAUTICAL_MILES)
            bigDF.iat[i,col4dist_diff]=distDelta
            # For later functionality...
            # if (timeDelta >= 60): 
            #     #avoiding divide by zero (small nums)
            #     aveSpeed=3600*distDelta/timeDelta
            #     bigDF.iat[i,col4aveSpeed]=aveSpeed
            #     #speed feature is knots, so need to convert time(secs) to hours
            # #else:  # not needed since preset to nan
            # #    bigDF.iat[i,col4aveSpeed]=np.nan
        else:
            if(flag_used_voyID):  #there was an actual voyage using last voyID
                voyID=voyID+1
                flag_used_voyID=False
    else:               #not at sea              
        if flag_atSea:  #only increment voyID if was last at sea
            voyID=voyID+1
        flag_atSea=False
        

print('After voyage ID assignments and stats, Dist statistics=:')
print(bigDF.dist_diff.describe())

#%%

print('After ship changes, Num NaN:',bigDF.dist_diff.isnull().sum())
print(bigDF.dist_diff.describe())

#store statistics
for s in shipTypes:
    # For voyage lenght metrics
    # condition=bigDF.type==s
    # sDF=bigDF.loc[condition,'time_diff']
    # norms.at[s,'time_diff_sd']=sDF.std()
    # norms.at[s,'time_diff_q75']=sDF.quantile(q=0.75)
    # norms.at[s,'time_diff_mean']=sDF.mean()
    
    #uncommented lines are a check on the difference
    #if calcs are done vi (voyage independent) since
    #that assignment is based on the assumption that within
    #500 meters of shore ends a voyage.
    
    # For distance metrics
    condition=bigDF.type==s
    sDF=bigDF.loc[condition,'dist_diff']
    # norms.at[s,'dist_diff_sd']=sDF.std()
    norms.at[s,'vi_dist_diff_q75']=sDF.quantile(q=0.75)
    # norms.at[s,'dist_diff_mean']=sDF.mean()

    # For time metrics
    condition=bigDF.type==s
    sDF=bigDF.loc[condition,'time_diff']
    # norms.at[s,'time_diff_sd']=sDF.std()
    norms.at[s,'vi_time_diff_q75']=sDF.quantile(q=0.75)
    # norms.at[s,'time_diff_mean']=sDF.mean()
    
print("*bigDF dist_diff=",bigDF.dist_diff.describe())
#%%

#Commented out lines for other metrics are future functionality

#Establish longest voyage for each ship
shipStats=pd.DataFrame()
voyageStats=pd.DataFrame()

#following also sets first column to voyID
voyageStats['mmsi']=bigDF.groupby('voyID').mmsi.min()
voyageStats['type']=bigDF.groupby('voyID').type.min()
voyageStats['voy_len']=bigDF.groupby('voyID').timestamp.max() \
                   -bigDF.groupby('voyID').timestamp.min()
voyageStats['numObs']=bigDF.groupby('voyID').mmsi.count()
voyageStats['ave_dist_gap']=bigDF.groupby('voyID').dist_diff.mean()
voyageStats['ave_time_gap']=bigDF.groupby('voyID').time_diff.mean()
#voyageStats['max_speed']=bigDF.groupby('voyID').speed.max()
#voyageStats['max_aveSpeed']=bigDF.groupby('voyID').aveSpeed.max()

#populate ship behavior across all voyages
shipStats['numVoy']=voyageStats.groupby('mmsi').type.count()
shipStats['mmsi']=shipStats.index
shipStats['numObsInVoys']=voyageStats.groupby('mmsi').numObs.sum()               
shipStats['type']=voyageStats.groupby('mmsi').type.agg(pd.Series.mode)
#! mode used in case re-typing of an mmsi.
shipStats['ave_len_voy']=voyageStats.groupby('mmsi').voy_len.mean()


#future!
#shipStats['max_len_voy']=voyageStats.groupby('mmsi').voy_len.max()
#shipStats['len_voy_q75']=voyageStats.groupby('type').voy_len.mean()

shipStats['ave_time_gap']=voyageStats.groupby('mmsi').ave_time_gap.mean()
shipStats['ave_dist_gap']=voyageStats.groupby('mmsi').ave_dist_gap.mean()
#shipStats['max_speed']=voyageStats.groupby('mmsi').max_speed.max()
#shipStats['max_aveSpeed']=voyageStats.groupby('mmsi').max_aveSpeed.max()

#now populate norms based on shipStats & voyageStats
# of voyages
norms['numVoyages']=voyageStats.groupby('type').mmsi.count()
#total length of voyages (seconds)
norms['voy_len_q75']=voyageStats.groupby('type').voy_len.quantile(q=0.75)
#time gaps between observations
norms['time_gap_q75']=voyageStats.groupby('type').ave_time_gap.quantile(q=0.75)
norms['dist_gap_q75']=voyageStats.groupby('type').ave_dist_gap.quantile(q=0.75)
#norms['time_gap_mean']=voyageStats.groupby('type').max_time_gap.mean()
#norms['time_gap_sd']=voyageStats.groupby('type').max_time_gap.std()
#norms['shipType']=norms.index

# #calculate risk columns (move to assess_risks)
# shipStats['risk_len_voy']=np.nan
# col4type=shipStats.columns.get_loc('type')
# col4risk=shipStats.columns.get_loc('risk_len_voy')
# # for s in shipTypes:
# #     #condition=norms.shipType==s
# #     limit=norms.at[s,'voy_len_q75']
# #     for s2 in range(0,len(shipStats)):
# #         if shipStats.iat[s2,col4type]==s:    
# #             shipStats['risk_len_voy']=shipStats.voy_lenvoy(limit


megaStats=shipStats.merge(norms, on='type',how='left')
print(norms.describe(include='all'))
print(shipStats.describe(include='all'))
print(voyageStats.describe(include='all'))

#%%
# export enough of bigDF to debug
first_mmsi=shipStats.iloc[0, 0]
print('first_mmsi=',first_mmsi)

condition=bigDF.mmsi==first_mmsi
testDF=bigDF.loc[condition,:]

#%%
# Export Results
path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_results"

norms.to_csv(os.path.join(path,'norms.csv'),na_rep='NULL')
shipStats.to_csv(os.path.join(path,'shipstats.csv'),na_rep='NULL')
voyageStats.to_csv(os.path.join(path,'voyageStats.csv'),na_rep='NULL')
megaStats.to_csv(os.path.join(path,'megaStats.csv'),na_rep='NULL')
testDF.to_csv(os.path.join(path,'testDF.csv'),na_rep='NULL')