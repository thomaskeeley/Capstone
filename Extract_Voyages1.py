# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 07:21:24 2020

@author: harry
"""

## Extract records where vessels are within 1km of shoreline
df=bigDF

shore = df.loc[df['distance_from_shore'] < 1000]
print(len(shore))

# Create a feature from the index to use for reference later
shore['saveIndex'] = shore.index

# Group by MMSI and view record numbers
shore_group = pd.DataFrame(shore.groupby(['mmsi'])['index'].nunique())

# Sort values of Shore data frame by datetime
shore = shore.sort_values(by=['zdatetime'])

#%%
# group by MMSI and calculate the time difference between coastal signal data
shore_time = pd.DataFrame(shore.groupby(['mmsi'])['zdatetime'].diff())
 
# Save the difference between time within 1km of shore as number of days
shore_time['days'] = pd.to_numeric(shore_time['zdatetime'].dt.days, downcast='integer')

# Change the column name for time difference
shore_time = shore_time.rename(columns={'zdatetime':'coastal_time_diff'})

#%%
# Extract all of the occurences of vessels being further than 1km from shore for more than 1 day
shore_time_ext = pd.DataFrame(shore_time.loc[shore_time['days'] > 0])
shore_time_ext['index'] = shore_time_ext.index

# Merge that with the original index values
shore_time_ext = pd.merge(shore_time_ext, shore[['index', 'mmsi', 'zdatetime']], on='index', how='left')

#%%

# Extract all of the signals where a vessel was further than 1km from shore for more than 100 days
susp = pd.DataFrame(shore_time_ext.loc[shore_time_ext['days'] >= 10])

#%%

# For every trip longer than 100 days, count the number of records between signal within 1km of shore
# A lot of the signals of ship signals show a ship being docked for a long time
# Look for the ships that have been >1km from shore for >100 days and have sent many signals in between

MMSI = []
start_index = []
start_date = []
end_index = []
end_date = []
voyage_time = []
voyage_signals = []
for i, series in susp.iterrows():
    index = series['index']
    mmsi = series['mmsi']
    days = series['days']
    end = series['zdatetime']
    
    count = 0
    data = df.loc[df['mmsi'] == mmsi].sort_values(by=['index'])
    data = data.loc[:index]
    
    for idx, records in data.iloc[::-1][1:].iterrows():
        if records['distance_from_shore'] > 1000:
            count += 1
        elif records['distance_from_shore'] <= 1000:
            break
    if count > 0:
        
        new_start_index = data.loc[idx+1]['index']
        new_start_date = data.loc[idx+1]['zdatetime']
        new_time_diff = end - new_start_date
        new_days = new_time_diff.days
        
        if new_days > 100:
            MMSI.append(mmsi)
            start_index.append(new_start_index)
            start_date.append(new_start_date)
            end_index.append(index)
            end_date.append(end)
            voyage_time.append(new_days)
            voyage_signals.append(count)
        
    voyage_df = pd.DataFrame(list(zip(MMSI, start_index, start_date, end_index, end_date, voyage_time, voyage_signals)), 
                             columns= ['mmsi', 'start_index', 'start_date', 'end_index', 'end_date', 'voyage_time', 'voyage_signals'])

    print(index, mmsi, days, new_time_diff, count)

#%%
# Choose and arbitrary vessel from that list to investigate in GIS
'''
# Save this as example of vessel being docked
susp1 = pd.DataFrame(df.loc[df['mmsi'] == '276111009888318'])

susp1.to_csv('susp1.csv')

susp2 = pd.DataFrame(df.loc[df['mmsi'] == '103912281609040'])
susp2.to_csv('susp2.csv')

for idx_, records in data.iloc[::-1][1:10].iterrows():
    print(idx_)


103912281609040
'''


209987524306087


74332170817758
