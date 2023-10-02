#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBRegressor
import xgboost as xgb


# In[2]:


def haversine_distance(origin, destination):
    """
    # Formula to calculate the spherical distance between 2 coordinates, with each specified as a (lat, lng) tuple

    :param origin: (lat, lng)
    :type origin: tuple
    :param destination: (lat, lng)
    :type destination: tuple
    :return: haversine distance
    :rtype: float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# In[20]:


def process_train_data(raw_df):
    """
    TODO: Implement this method.
    
    You may drop rows if needed.

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    #filter range-north america
    raw_df = raw_df.loc[raw_df.pickup_latitude.between(10.1,84.0) & raw_df.pickup_longitude.between(-175.6,-8.0)]
    #filter fare amount
    raw_df = raw_df[raw_df['fare_amount'] < 79.6558]
    #calculate distance
    raw_df['dropoff'] = list(zip(raw_df['dropoff_latitude'], raw_df['dropoff_longitude']))
    raw_df['pickup'] = list(zip(raw_df['pickup_latitude'], raw_df['pickup_longitude']))
    raw_df['distance'] = raw_df.apply(lambda row: haversine_distance(row['dropoff'], row['pickup']), axis = 1)
    #drop intermediate step columns for calculating distance
    raw_df = raw_df.drop(['dropoff','pickup'], axis=1)
    #convert datetime to numbers
    raw_df['year'] = raw_df.pickup_datetime.dt.year
    raw_df['month'] = raw_df.pickup_datetime.dt.month
    raw_df['hour'] = raw_df.pickup_datetime.dt.hour
    raw_df['weekday'] = raw_df.pickup_datetime.dt.weekday
    return raw_df


def process_test_data(raw_df):
    """
    TODO: Implement this method.
    
    You should NOT drop any rows.

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    #calculate distance
    raw_df['dropoff'] = list(zip(raw_df['dropoff_latitude'], raw_df['dropoff_longitude']))
    raw_df['pickup'] = list(zip(raw_df['pickup_latitude'], raw_df['pickup_longitude']))
    raw_df['distance'] = raw_df.apply(lambda row: haversine_distance(row['dropoff'], row['pickup']), axis = 1)
    #drop intermediate step columns for calculating distance
    raw_df = raw_df.drop(['dropoff','pickup'], axis=1)
    #convert datetime to numbers
    raw_df['year'] = raw_df.pickup_datetime.dt.year
    raw_df['month'] = raw_df.pickup_datetime.dt.month
    raw_df['hour'] = raw_df.pickup_datetime.dt.hour
    raw_df['weekday'] = raw_df.pickup_datetime.dt.weekday
    return raw_df


# In[7]:


# Load data
raw_train = pd.read_csv('data/cc_nyc_fare_train_small.csv', parse_dates=['pickup_datetime'])
print('Shape of the raw data: {}'.format(raw_train.shape))


# In[21]:


# Transform features using the function you have defined
df_train = process_train_data(raw_train)

# Remove fields that we do not want to train with
X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1, errors='ignore')

# Extract the value you want to predict
Y = df_train['fare_amount']
print('Shape of the feature matrix: {}'.format(X.shape))


# In[25]:


# Build final model with the entire training set
final_model = XGBRegressor(objective ='reg:squarederror')
final_model.fit(X, Y)

# Read and transform test set
raw_test = pd.read_csv('data/cc_nyc_fare_test.csv', parse_dates=['pickup_datetime'])
df_test = process_test_data(raw_test)
X_test = df_test.drop(['key', 'pickup_datetime'], axis=1, errors='ignore')

# Make predictions for test set and output a csv file
# DO NOT change the column names
df_test['predicted_fare_amount'] = final_model.predict(X_test)
df_test[['key', 'predicted_fare_amount']].to_csv('predictions.csv', index=False)


# In[ ]:




