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
#"drifting_longlines" #, for debugging picking just one
#"fixed_gear",
#"pole_and_line",
#"purse_seines",
"trawlers"
#"trollers",
#"unknown"
]
columnNames=['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port',
       'speed', 'course', 'lat', 'lon', 'is_fishing', 'source']
dataTypes={'mmsi':float,'timestamp':object,'distance_from_shore':float, \
            'distance_from_port':float, 'speed':float, 'course':float, \
            'lat':float,'lon':float, 'is_fishing':object, 'source':object}
    
bigDF=pd.DataFrame(columns=columnNames)#,dtype=dataTypes)

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
voyStatCols=['numObs','numShips','meanVoyage','sdVoyage','numVoyage']
voyStat=pd.DataFrame(columns=voyStatCols,dtype=object)

#Determine number of observations (signals) and unique ships for each ship type 
voyStat.numObs=bigDF.groupby("type")["mmsi"].count()

voyStat.numShips=bigDF.groupby("type")["mmsi"].nunique()

#%%
# Prepare for risk factors on gaps in ship signals (both time & distance)

#Make sure observations are ordered first by ship then time observed
bigDF=bigDF.sort_values(['mmsi','timestamp'])

#This may be needed but I think that I can avoid the compute
# Convert UNIX timestamp to date-time and calculate intervals between signal transmissions
#bigDF['ztimestamp'] = pd.to_datetime(bigDF['timestamp'],unit='s')
#don't think that I need to convert since its already in seconds
bigDF['timestamp']=bigDF['timestamp'].astype(float)

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
    voyStat.at[s,'time_diff_sd']=sDF.std()
    voyStat.at[s,'time_diff_q']=sDF.quantile(q=0.75)
    voyStat.at[s,'time_diff_mean']=sDF.mean()

#Plan to use the q=.75 but will run by SMEs. 
#%%    
# Determine distance interval
#For now, use haverstine on each pair of observations for dist travelled
#https://pypi.org/project/haversine/ 
#Later, will optimize using S2 library by Google
#Can also apply logic to use dist to shore as a surrogate if this proves
#too costly since that was already computed in the dataset


bigDF['dist_diff']=np.nan
col4diff=bigDF.columns.get_loc('dist_diff')
col4lat=bigDF.columns.get_loc('lat')
col4lon=bigDF.columns.get_loc('lon')
for i in range(1,len(bigDF)):
    bigDF.iat[i,col4diff] = \
        haversine((bigDF.iat[i,col4lat],bigDF.iat[i,col4lon]), \
                    (bigDF.iat[i-1,col4lat],bigDF.iat[i-1,col4lon]))
        
# Eliminate meaningless intervals when the prev row was for a different ship
condition=bigDF.mmsi!=bigDF.mmsi.shift(1)
bigDF.loc[condition,'dist_diff']=np.nan

print('After ship changes, Num NaN:',bigDF.dist_diff.isnull().sum())
print(bigDF.dist_diff.describe())
#%%



#%%

