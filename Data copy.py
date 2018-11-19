#!/usr/bin/env python
# coding: utf-8

# In[108]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sq
import time
import math
import re
from sklearn import preprocessing


# In[109]:


#just reading data
con = sq.connect("database.sqlite")
team_atts = pd.read_sql_query("SELECT * from Team_Attributes", con)
teams = pd.read_sql_query("SELECT * from Team", con)
matches = pd.read_sql_query("SELECT * from Match", con)
matches = matches[['date', 'home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id', 
                  'goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession',
                  'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD',
                  'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']]


# In[110]:


# print(team_atts.columns.values)

### Tasks TODO: 
### Drop columns that are redundant
### Drop rows that are missing too many values
### Replace text data such as "slow, medium, fast" with number mappings that make sense (e.g. 0, 1, 2) 
### Normalize all features to be in the range (0, 1): subtract the low value and divide by (high - low)

buildUpPlaySpeed = {'Slow': 0, 'Balanced': 1, 'Fast': 2}

st = time.time()

denom = len(matches.columns.values)


### Add blank columns for team attributes to be filled in for each match
for column in list(team_atts.columns.values):
    matches['__home_' + column] = np.nan
    
for column in list(team_atts.columns.values):
    matches['__away_' + column] = np.nan

### To assist in filling values later (note the underscores leading __underscoes added above & used here 
### so we don't collide with existing column names)
home_column_indexes = [matches.columns.get_loc('__home_' + col_name) for col_name in team_atts.columns.values]
away_column_indexes = [matches.columns.get_loc('__away_' + col_name) for col_name in team_atts.columns.values]
indexes_to_drop = []

## Part of experiments described below
n_15_none_match = 0
n_15_none_team_att = 0

for index, match in matches.iterrows():
    ### For each match, we find the home and away team for the correct year, and add their data to the 
    ### dataframe
    year = match['date'][:4]
    home_team_id = match['home_team_api_id']
    away_team_id = match['away_team_api_id']
    home_team_atts = team_atts.loc[team_atts['team_api_id'] == home_team_id]
    away_team_atts = team_atts.loc[team_atts['team_api_id'] == away_team_id]
    home_team_att = home_team_atts.loc[team_atts['date'].str.contains(year)]
    away_team_att = away_team_atts.loc[team_atts['date'].str.contains(year)]
    
    
    ### This is just an experiment to determine a threshold for how many values should be 'None'
    ### in match data in order for us to drop a row. To drop a row, add its index to "indexes_to_drop"
    ### if too many values are 'None'
    pct_match_none = sum(1 for val in match.values if val is None) / denom
    if pct_match_none > 0.15:
        n_15_none_match += 1

    if not home_team_att.empty and not away_team_att.empty:
        matches.iloc[index, home_column_indexes] = home_team_att.values[0]
        matches.iloc[index, away_column_indexes] = away_team_att.values[0]
        
        
        ### This is just an experiment to determine a threshold for how many values should be 'None'
        ### in team attribute data in order for us to drop a row. To drop a row, add its index to "indexes_to_drop"
        ### if too many values are 'None'
        pct_home_none = sum(1 for val in home_team_att.values[0] if val is None) / len(home_team_att.values)
        pct_away_none = sum(1 for val in away_team_att.values[0] if val is None) / len(away_team_att.values)
        if pct_home_none > 0.15 or pct_away_none > 0.3:
            n_15_none_team_att += 1
        
    else:
        indexes_to_drop.append(index)

### Part of our experiments
n_rows = index
print('total input rows:', n_rows)
print('num lacking any team attribute data:', len(indexes_to_drop))
print('num where >15% of team data is None:', n_15_none_match)
print('num where >15% of team attribute data is None:', n_15_none_team_att)

matches = matches.drop(indexes_to_drop, axis=0) ### Drops rows that lack too much data

print('Took {0:.2f} seconds.'.format(time.time() - st))
matches


# In[111]:


matches.to_csv('data_step_1.csv')


# In[112]:


matches


# In[113]:


print(matches.columns.values, len(matches.columns.values))

#drop first 13 of matches:
matches = matches.drop(['date', 'home_team_goal', 'away_team_goal' ,'home_team_api_id',
 'away_team_api_id', 'goal', 'shoton' ,'shotoff', 'foulcommit', 'card', 'cross',
 'corner', 'possession'], axis=1)


# In[114]:


print(matches.columns.values, len(matches.columns.values))
print(matches)

            


# In[115]:


matches


# In[116]:


#Enumerate the columns if they have string values
newCol = {}
for col in matches.columns.values:
    if re.search('Class', col):
            #print(col, matches[col])
            #enum_dict = dict(enumerate(list(set(matches[col]))))
            enum_dict = { k: v for v, k in dict(enumerate(list(set(matches[col])))).items()}
            #print(col, enum_dict)
            #print(matches[col])
            newCol[col] = matches[col].map(enum_dict)
#print(newCol['__home_buildUpPlaySpeedClass'])
for colName in newCol.keys():
    matches[colName] = newCol[colName]
matches.to_csv('data_enumerated.csv')
matches


# In[117]:


# __home_buildUpPlaySpeedClass {'Balanced': 0, 'Slow': 1, 'Fast': 2}
# __home_buildUpPlayDribblingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __home_buildUpPlayPassingClass {'Mixed': 0, 'Long': 1, 'Short': 2}
# __home_buildUpPlayPositioningClass {'Organised': 0, 'Free Form': 1}
# __home_chanceCreationPassingClass {'Risky': 0, 'Normal': 1, 'Safe': 2}
# __home_chanceCreationCrossingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __home_chanceCreationShootingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __home_chanceCreationPositioningClass {'Organised': 0, 'Free Form': 1}
# __home_defencePressureClass {'Deep': 0, 'Medium': 1, 'High': 2}
# __home_defenceAggressionClass {'Double': 0, 'Contain': 1, 'Press': 2}
# __home_defenceTeamWidthClass {'Normal': 0, 'Wide': 1, 'Narrow': 2}
# __home_defenceDefenderLineClass {'Offside Trap': 0, 'Cover': 1}
# __away_buildUpPlaySpeedClass {'Balanced': 0, 'Slow': 1, 'Fast': 2}
# __away_buildUpPlayDribblingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __away_buildUpPlayPassingClass {'Mixed': 0, 'Long': 1, 'Short': 2}
# __away_buildUpPlayPositioningClass {'Organised': 0, 'Free Form': 1}
# __away_chanceCreationPassingClass {'Normal': 0, 'Risky': 1, 'Safe': 2}
# __away_chanceCreationCrossingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __away_chanceCreationShootingClass {'Normal': 0, 'Little': 1, 'Lots': 2}
# __away_chanceCreationPositioningClass {'Organised': 0, 'Free Form': 1}
# __away_defencePressureClass {'Deep': 0, 'Medium': 1, 'High': 2}
# __away_defenceAggressionClass {'Double': 0, 'Contain': 1, 'Press': 2}
# __away_defenceTeamWidthClass {'Normal': 0, 'Wide': 1, 'Narrow': 2}
# __away_defenceDefenderLineClass {'Offside Trap': 0, 'Cover': 1}


# In[118]:


# Get rid of cols missing betting odds
to_remove = []
no_missing = matches.columns.values[:30]
for index, match in matches.iterrows():
    for col in no_missing: 
        if pd.isnull(match[col]):
            to_remove.append(index)
matches = matches.drop(to_remove, axis=0)
    


# In[131]:


new_to_remove = []
for col in matches.columns.values:
    if re.search('date', col):
        new_to_remove.append(col)
matches = matches.drop(new_to_remove, axis=1)


# In[132]:


# #TODO: Normalize columns
# ###################
# for 
#     x = df[['score']].values.astype(float)

#     # Create a minimum and maximum processor object
#     min_max_scaler = preprocessing.MinMaxScaler()

#     # Create an object to transform the data to fit minmax processor
#     x_scaled = min_max_scaler.fit_transform(x)

#     # Run the normalizer on the dataframe
#     df_normalized = pd.DataFrame(x_scaled)


# In[133]:


matches.to_csv("full_betting_odds.csv")


# In[134]:


#fill in missing data with na with -1
#CHANGE LATER TO BE MORE ROBUST
matches = matches.fillna(-1)
matches.to_csv('negative_one_fill.csv')
matches


# In[135]:


#get labels
new_to_remove = []
og = pd.read_sql_query("SELECT * from Match", con)
index = range(0,og.shape[0]) # number rows
columns = ['Home', 'Draw', 'Away']
labels =  pd.DataFrame(index=index, columns=columns)
print(index, matches.index.values)
for index, match in og.iterrows():
    if index in matches.index.values:
        if match['home_team_goal'] > match['away_team_goal']:
            labels.at[index, 'Home'] = 1
        elif match['home_team_goal'] == match['away_team_goal']:
            labels.at[index, 'Draw'] = 1
        else:
            labels.at[index, 'Away'] = 1
    else:
        new_to_remove.append(index)
labels = labels.drop(new_to_remove, axis=0)
labels = labels.fillna(0)
print(labels, labels.shape[0] == matches.shape[0])
labels.to_csv('labels.csv')


# In[136]:


#shuffle match rows so split tables are randomized
# matches = matches.reindex(np.random.permutation(matches.index))

matches.to_csv('cleaned_data.csv')
#split match data into training, validation, and test sets
# m_train = matches.iloc[:17861]
# m_valid = matches.iloc[17861:21108]
# m_test = matches.iloc[21108:]


# In[ ]:





# In[ ]:





# In[ ]:




