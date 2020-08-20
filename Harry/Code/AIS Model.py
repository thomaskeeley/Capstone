# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 05:33:40 2020

@author: harry
"""

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


#%% Global Variables
path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_data"

    
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
       'speed', 'course', 'lat', 'lon', 'is_fishing', 'source','type','voyID', \
       'time_diff','dist_diff']
dataTypes={'mmsi':float,'timestamp':float,'distance_from_shore':float, \
             'distance_from_port':float, 'speed':float, 'course':float, \
             'lat':float,'lon':float, 'is_fishing':object, 'source':object, \
             'type':str, 'voyID':int, 'time_diff':float, 'dist_diff':float }
dataUnits={'mmsi':'VesselID','timestamp':'seconds','distance_from_shore':'meters', \
            'distance_from_port':'meters', 'speed':'knots', 'course':'degrees', \
            'lat':'decimal degrees','lon':'decimal degrees', \
            'is_fishing':'label', 'source':'text','type':'text','voyID':'int', \
            'time_diff':'seconds','dist_diff':'nm'}

#bigDF will hold the AIS data (millions of rows)
bigDF=pd.DataFrame(columns=columnNames) #,dtype=dataTypes)

#testDF will have the same columns as bigDF but few enough rows to debug with
testDF=pd.DataFrame(columns=columnNames) #,dtype=dataTypes)

#norms will have one row per ship type
normsCols=['numObs','numShips','numVoyages','aveNumVoyages','numVoyages_q25']
norms=pd.DataFrame(columns=normsCols,dtype=object)

#voyageStats will have one row per voyage per ship (1000s of rows)
voyageStats=pd.DataFrame()

#shipStats will have one row per ship (1000s)
shipStats=pd.DataFrame()

#megaStats--will have one row pership and cols from shipStats and norms
# so that risk factors can be assessed directly from a row without other lookups

megaStats=pd.DataFrame()

#%% read_AIS
def read_AIS(st):
    #Read csv file for "st" shipType and return dataframe
    #assumes files named for each ship type. 
    
    df = pd.read_csv(os.path.join(path,st+".csv"),dtype=dataTypes)
    print("Read: ", st, "Records: ",len(df)); 
    df['type']=st
         #   df['mmsi'] = df['mmsi'].astype(int)
         #   print("Features: ",df.columns)
             
    return(df)

        
#%% assign voyages

def assign_voyages(breakPointForPortCall):
   
    # Determine distance between signals for each voyage of each ship
    #For now, use haverstine on each pair of observations for dist travelled
    #https://pypi.org/project/haversine/ 
    #Later, could optimize using S2 library or
    #distance_to_shore as a surrogate if this proves
    #too costly since that was already computed in the dataset
    
    #Will assign voyage unique IDs while computing distances
    #with logic that within 500 meters of shore  is not a voyage 
    
    print("in assign_voyID, breakpoint=",breakPointForPortCall)
    
    #reference columns to compute time and distance deltas
    col4dist2shore=bigDF.columns.get_loc('distance_from_shore')
    col4dist2port=bigDF.columns.get_loc('distance_from_port')
    
    #setup to label each voyage
    bigDF['voyID']=0
    col4voyID=bigDF.columns.get_loc('voyID')
    col4mmsi=bigDF.columns.get_loc('mmsi')
    
    #Assign first row since it can't be computed with delta formulas
    voyID=1  #each voyage will get a unique ID
    flag_used_voyID=False
    
    i=0
    dist2port=bigDF.iat[i,col4dist2port]

    if (dist2port>=breakPointForPortCall):
        flag_atSea=True
        bigDF.iat[i,col4voyID]=voyID
        flag_used_voyID=True
        
    #Starting index at 1 because the 0th row has no i-1 row for deltas.
    flag_atSea=False
    for i in range(1,len(bigDF)):
        dist2port=bigDF.iat[i,col4dist2port]
        #if at sea & same ship, compute distance and assign voyageID
        if (dist2port>breakPointForPortCall): #at sea
            flag_atSea=True
            if (bigDF.iat[i,col4mmsi] == bigDF.iat[i-1,col4mmsi]): #same ship
                bigDF.iat[i,col4voyID]=voyID
                flag_used_voyID=True
                
            else:
                if(flag_used_voyID):  #there was an actual voyage using last voyID
                    voyID=voyID+1
                    bigDF.iat[i,col4voyID]=voyID
                    flag_used_voyID=True
        else:               #not at sea              
            if flag_atSea:  #only increment voyID if was last at sea
                voyID=voyID+1
            flag_atSea=False
            
    print('In voyage assignments, statistics=:')
    print(bigDF.voyID.describe())
    
    
    
    #drop_bad_voyages()
    
    return()

#%% compute_deltas
#after have quality voyages, compute deltas for each ais within voyage
def compute_deltas(): 
    # Determine distance between signals for each voyage of each ship
    #For now, use haverstine on each pair of observations for dist travelled
    #https://pypi.org/project/haversine/ 
    #Later, could optimize using S2 library or
    #distance_to_shore as a surrogate if this proves
    #too costly since that was already computed in the sample dataset
     
    #setup for summary stats between each observation    
    bigDF['time_diff']=np.nan
    bigDF['dist_diff']=np.nan

    #setup for results in time and dist diff 
    col4time_diff=bigDF.columns.get_loc('time_diff')
    col4dist_diff=bigDF.columns.get_loc('dist_diff')
    
    #reference columns to compute time and distance deltas
    col4lat=bigDF.columns.get_loc('lat')
    col4lon=bigDF.columns.get_loc('lon')
    col4timestamp=bigDF.columns.get_loc('timestamp')
   
    col4voyID=bigDF.columns.get_loc('voyID')
    col4mmsi=bigDF.columns.get_loc('mmsi')
    lastmmsi=bigDF.iat[0,col4mmsi]
    
    last_voyID=-1
    #Starting index at 1 because the 0th row has no i-1 row for deltas.
    for i in range(0,len(bigDF.index)-1):
        voyID=bigDF.iat[i,col4voyID]
        if (last_voyID==voyID):
            if(lastmmsi != bigDF.iat[i,col4mmsi]): #debug
                print("**error in compute_diff, swapped mmsi within voyage")
                print("lastmmsi=",lastmmsi,"i=",i," mmsi=",bigDF.iat[i,col4mmsi])
            #compute time delta
            timeDelta=bigDF.iat[i,col4timestamp]-bigDF.iat[i-1,col4timestamp]
            bigDF.iat[i,col4time_diff]=timeDelta
            
            #compute distance delta
            distDelta=haversine((bigDF.iat[i,col4lat],bigDF.iat[i,col4lon]), \
                              (bigDF.iat[i-1,col4lat],bigDF.iat[i-1,col4lon]),\
                               unit=Unit.NAUTICAL_MILES)
            bigDF.iat[i,col4dist_diff]=distDelta
            
        else:
            # i is the first row of a new voyage do set deltas to NaN
            bigDF.iat[i,col4time_diff]=np.NaN
            bigDF.iat[i,col4dist_diff]=np.NaN
            last_voyID=voyID
            lastmmsi=bigDF.iat[i,col4mmsi]
    
            
    print('In compute_diffs statistics=:')
    print(bigDF.dist_diff.describe())
    print(bigDF.time_diff.describe())
    
    # export enough of bigDF to debug
    first_mmsi=bigDF.iat[0, col4mmsi]
    print('first_mmsi=',first_mmsi)
    
    testSample_condition=bigDF.mmsi==first_mmsi
    testDF=bigDF.loc[testSample_condition,:]
    write_sample(first_mmsi, testDF, 'test_after_voyages_assigned')
    
    return()

#%% voyage_stats
def voyage_statistics():
    print('After ship changes, Num NaN:',bigDF.dist_diff.isnull().sum())
    print(bigDF.dist_diff.describe())
    
    #store statistics
    #voyage stats
    
    print("big DF shape",bigDF.shape)
    #Commented out lines for other metrics are future functionality
    condition=(bigDF.time_diff>1.0) & (bigDF.dist_diff>1.0) & (bigDF.voyID>0)
    validDF=bigDF.loc[condition,:]
    print("valid DF shape",validDF.shape)
    
    #following also sets first column to voyID
    voyageStats['mmsi']=bigDF.groupby('voyID').mmsi.min()
    voyageStats['type']=bigDF.groupby('voyID').type.max()
    voyageStats['numObs']=bigDF.groupby('voyID').mmsi.count()
    voyageStats['voyID']=voyageStats.index
    
    #voyageStats['max_speed']=bigDF.groupby('voyID').speed.max()
    #voyageStats['max_aveSpeed']=bigDF.groupby('voyID').aveSpeed.max()
    return()

#%% Drop voyages
#Drop bad voyages and associated data
#drop voyages with only one observation since those 
# are not useful for statistics and on examination of the data 
# appear to be caused by data errors.  It's not attempted spoofing
# because it only occurs for one blip.
def drop_bad_voyages():
    saveBigDF=bigDF
    memVoyageStats=voyageStats
    print("Voyage Stats Shape: ")
    print(voyageStats.shape)
    condition=voyageStats.numObs==1
    #savelist for bigDF surgery
    voyagesRemoved=voyageStats.loc[condition,'voyID']
    
    print("Dropping: ",sum(condition))
    voyageStats.drop(voyageStats[condition].index,inplace=True)
    print("Voyage Stats Shape: ")
    print(voyageStats.shape)
    
    #Now drop the corresponding entries from bigDF...
    print("before big drop, shape=")
    print(bigDF.shape)
    VRlist=voyagesRemoved.to_list()
    bigDF['drop_it']=bigDF.voyID.isin((VRlist))
    condition=bigDF.drop_it==True
    bigDF.drop(bigDF[condition].index,inplace=True)
    print("after big drop, shape=")
    print(bigDF.shape)
    
    return



    
#%% Populate ship behavior Edge Tables across all voyages
def ship_statistics():
    condition=(bigDF.voyID>0)
    sDF=bigDF.loc[condition,:]

    voyageStats['ave_dist_gap']=sDF.groupby('voyID').dist_diff.mean()
    voyageStats['ave_time_gap']=sDF.groupby('voyID').time_diff.mean()
    voyageStats['starttime']=sDF.groupby('voyID').timestamp.min()
    voyageStats['stoptime']=sDF.groupby('voyID').timestamp.min()
    voyageStats['voy_len']=voyageStats['stoptime']-voyageStats['starttime']
    voyageStats['old_voy_len']=bigDF.groupby('voyID').timestamp.max() \
                   -bigDF.groupby('voyID').timestamp.min()
    shipStats['numVoy']=voyageStats.groupby('mmsi').voy_len.count()
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
    
    #populate voyage independent shipStats
    # For distance metrics
    shipStats['vi_dist_gap']=bigDF.groupby('mmsi').dist_diff.mean()
    
    # For time metrics
    shipStats['vi_time_gap']=bigDF.groupby('mmsi').time_diff.mean()
    
    
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
    
    #Note, may need to ".reset_index()" on some of these df funcs
    
    return()    

#%% define normal behaivor

def normal_behavior():
    #store statistics for each ship type
    for s in shipTypes:

        # For distance metrics
        condition=(bigDF.type==s) & (bigDF.voyID>0)
        sDF=bigDF.loc[condition,'dist_diff']
        # norms.at[s,'dist_diff_sd']=sDF.std()
        norms.at[s,'vi_dist_diff_q75']=sDF.quantile(q=0.75)
        # norms.at[s,'dist_diff_mean']=sDF.mean()
    
        # For time metrics
        condition=(bigDF.type==s) & (bigDF.voyID>0)
        sDF=bigDF.loc[condition,'time_diff']
        # norms.at[s,'time_diff_sd']=sDF.std()
        norms.at[s,'vi_time_diff_q75']=sDF.quantile(q=0.75)
        # norms.at[s,'time_diff_mean']=sDF.mean()
        
        print("*bigDF dist_diff=",bigDF.dist_diff.describe())
        print("*bigDF time_diff=",bigDF.time_diff.describe())
  
    #Plan to use the q=.75 but will run by SMEs.
    
    #Determine number of observations (signals) and unique ships for each ship type 
    norms.numObs=bigDF.groupby("type")["mmsi"].count()
    
    norms.numShips=bigDF.groupby("type")["mmsi"].nunique()
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
    return()
#%% Export Results
# Export Results
def export_results():
    path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_results"
    
    norms.to_csv(os.path.join(path,'norms.csv'),na_rep='NULL')
    shipStats.to_csv(os.path.join(path,'shipstats.csv'),na_rep='NULL')
    voyageStats.to_csv(os.path.join(path,'voyageStats.csv'),na_rep='NULL')
    megaStats.to_csv(os.path.join(path,'megaStats.csv'),na_rep='NULL')
    testDF.to_csv(os.path.join(path,'testDF.csv'),na_rep='NULL')
    return()
#%%  write sample
# Function to write subset of data for debugging.  
def write_sample(mmsi, df, filename):
    print("Exporting Sample to "+filename+" for mmsi="+str(mmsi)+" of size ")
    print(df.shape)
    path = "C:\\Users\\harry\\Github\\CAP\\Capstone\\AIS_results"
    testDF.to_csv(os.path.join(path,filename),na_rep='NULL')
    return()

#%% main
    
#READ AIS DATA AND PROVIDE SAMPLE
for s in shipTypes:  ##[2:4] to restrict for debugging
   bigDF=bigDF.append(read_AIS(s),ignore_index=True)
            
#Make sure observations are ordered first by ship then time observed
bigDF=bigDF.sort_values(['mmsi','timestamp'])

# export enough of bigDF to debug
first_mmsi=bigDF.iloc[0, 0]
testSample_condition=bigDF.mmsi==first_mmsi
testDF=bigDF.loc[testSample_condition,:]
write_sample(first_mmsi, testDF, 'test_after_read')

#Assign voyages based on logic of proximity to a port breaks voyages

assign_voyages(breakPointForPortCall=200)
#Compute stats over a voyage
voyage_statistics()
drop_bad_voyages()

assign_voyages(breakPointForPortCall=200)

#Within each voyage compute time & distance deltas
#  add drop & merge voyage logic later
compute_deltas()

#Compute stats for each ship
ship_statistics()

normal_behavior()

# prepare "megastats" with the ships behaivor and normals for the type
# this will be used to asses risks and display trends in other codes

norms['type']=norms.index
megaStats=shipStats.merge(norms, on='type',how='left')

#Report general statistics fo the important tables
print(norms.describe(include='all'))
print(shipStats.describe(include='all'))
print(voyageStats.describe(include='all'))

export_results()
print ("That's all folks...")
