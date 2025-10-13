#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from pickle import load

# In[2]:


# ## Load Model
model = load_model('./output/LSTM_model.h5')

# ## Load Scalers
# ### Save The Scalers For Demand AND Temporarture
td_transformer = load(open('./output/scaler_td.pkl', 'rb'))
temp_transformer = load(open('./output/scaler_temperature.pkl', 'rb'))


# ## Import Data

# In[3]:


df = pd.read_csv('../../data/data_for_forecast.csv')

df.head()


# prepare the categoorical inputs with one hot encoding

# ## One Hot Encoding

# We have to do this manually, as the required forecast will not cover all the months, and possibly not all the time slots.


# ### Month & Hour

# In[4]:
records = len(df)

month_col = ['MONTH_1', 'MONTH_2', 'MONTH_3', 'MONTH_4', 'MONTH_5', 'MONTH_6', 'MONTH_7', 'MONTH_8', 'MONTH_9', 'MONTH_10', 'MONTH_11', 'MONTH_12']     


month = pd.DataFrame(np.zeros((records, len(month_col)), dtype=int), columns = month_col)


hour_col =['HOUR_0', 'HOUR_1', 'HOUR_2', 'HOUR_3', 'HOUR_4', 'HOUR_5', 'HOUR_6', 'HOUR_7', 'HOUR_8', 'HOUR_9', 'HOUR_10', 'HOUR_11','HOUR_12','HOUR_13','HOUR_14','HOUR_15', 'HOUR_16','HOUR_17','HOUR_18','HOUR_19', 'HOUR_20', 'HOUR_21','HOUR_22','HOUR_23']

hour = pd.DataFrame(np.zeros((records, len(hour_col)), dtype=int), columns = hour_col)

# fill in appropriately
for i in range(records):
    m = df.at[i,'MONTH']
    month.iat[i, m-1] = 1
    h = df.at[i, 'HOUR']
    hour.iat[i, h] = 1


# ## Scale Continuous Variables


# ### Temperature

# In[5]:
temperature = pd.DataFrame();
temperature['TEMPERATURE'] = df['TEMPERATURE']

temperature['TEMPERATURE'] = temp_transformer.transform(temperature[['TEMPERATURE']])


# ### Total Demand

# In[6]:
demand = pd.DataFrame()
demand['TOTALDEMAND'] = df['TOTALDEMAND']

demand['TOTALDEMAND'] = td_transformer.transform(demand[['TOTALDEMAND']])

# ## Build Required Data

# In[7]:
X = pd.concat([month, hour, df.WEEKEND, df.PUBLIC_HOLIDAY, temperature, demand ], axis=1)


# ## Configure Lookback Period And Apply To Dataset

# In[8]:


def create_dataset(X, time_steps=1):
    Xs = []
    v = X.iloc[0:time_steps].values
    Xs.append(v)        
        
    return np.array(Xs)

# ##Algorithm - Prediction one at a time

#
# Pass the time_steps number of records to predict the next forecast.
# Update the forecasted demand as actual demand for the relevant record.
# Drop the first record, and send the next time_steps number of records.
# Continue this until all prediction done.

#
# Precondition, that the original data file feeded should have time_steps number of records
# with actual demand
#

# ### Predict the demand one at a time
# In[9]:

time_steps = 24

# collect all the prediction 
df_pred = pd.DataFrame()
df_pred = pd.DataFrame(np.zeros((records-time_steps, 1)), columns = ['PREDICTION'])


for i in range(records - time_steps):
    X_batch = X.head(time_steps)
    X_test  = create_dataset(X_batch, time_steps)
    
    # predict    
    y_pred = model.predict(X_test)      
    df_pred.at[i,'PREDICTION']= y_pred
    
    #
    # Update total demand with the predicted value
    # And drop the oldest record for next prediction
    #
    X.at[time_steps,'TOTALDEMAND'] = y_pred
    X = X.tail(-1)
    X.reset_index(drop=True, inplace=True)

print(df_pred.head())

# ## Save The Prediction.

# In[10]:
    



# reset the index as it is a filtered set
df.reset_index(drop=True, inplace=True) 

# drop the records used for the first prediction
df.drop(range(0,time_steps), axis = 0, inplace=True) 

# reset the index as we dropped records.
df.reset_index(drop=True, inplace=True) 
KeyData = ['DATETIME', 'TEMPERATURE']

# rescale the forecasted demand
y_pred = td_transformer.inverse_transform(df_pred)
y_pred = y_pred.reshape(-1)
y_pred = pd.Series(y_pred)
y_pred = y_pred.to_frame()
y_pred.columns =['PREDICTION']

df_out = pd.concat([df[KeyData], y_pred], axis=1)

print(df_out.tail())
df_out.to_csv('./output/LSTM_predictions.csv', index = False)

