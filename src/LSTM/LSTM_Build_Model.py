#!/usr/bin/env python
# coding: utf-8

# ## Load Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

from pickle import dump


# ## Import Data

# In[2]:


df = pd.read_csv('../../data/merged_all_values_v2.csv')


# In[3]:


df.head()


# ## Take Subset Of Data To Only Include Years 2017, 2018, 2019 and 2020

# In[4]:


df = df[(df.YEAR == 2018) | (df.YEAR == 2019) | (df.YEAR == 2020) | (df.YEAR == 2021)]

# prepare the categoorical inputs with one hot encoding

# ## One Hot Encoding

# ### Month

# In[5]:


month = pd.get_dummies(df.MONTH, prefix='MONTH')



# ### Hour

# In[6]:


hour = pd.get_dummies(df.HOUR, prefix='HOUR')

# ## Scale Continuous Variables

# ### Temperature

# In[7]:
temperature = pd.DataFrame();
temperature['TEMPERATURE'] = df['TEMPERATURE']

temp_transformer = MinMaxScaler(feature_range=(0, 1))
temp_transformer = temp_transformer.fit(temperature[['TEMPERATURE']])
temperature['TEMPERATURE'] = temp_transformer.transform(temperature[['TEMPERATURE']])


# ### Total Demand

# In[8]:
demand = pd.DataFrame()
demand['TOTALDEMAND'] = df['TOTALDEMAND']
td_transformer = MinMaxScaler(feature_range=(0, 1))
td_transformer = td_transformer.fit(demand[['TOTALDEMAND']])

demand['TOTALDEMAND'] = td_transformer.transform(demand[['TOTALDEMAND']])

# ## Prepare the Input Series - using separate variable, so that we can save the result with original timestamp

# In[9]:
X = pd.concat([df.YEAR, month, hour, df.WEEKEND, df.PUBLIC_HOLIDAY, temperature, demand ], axis=1)

# ## Create Train and Test Datasets
train = X[X.YEAR != 2021]
test = X[X.YEAR == 2021]




# In[10]:
# ## Drop Columns Not Required
train.drop(['YEAR'], axis = 1, inplace= True)
test.drop(['YEAR'], axis = 1, inplace= True)




# ## Configure Lookback Period And Apply To Train And Test Datasets

# In[11]:


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# In[12]:
print(test.head())

time_steps = 24

# Prepare The Train And Test Sets

X_train, y_train = create_dataset(train, train.TOTALDEMAND, time_steps)
X_test, y_test = create_dataset(test, test.TOTALDEMAND, time_steps)

print(X_train.shape, y_train.shape)



# ## Configure LSTM Neural Network

# In[13]:


model = Sequential()

model.add(
    Bidirectional(LSTM(units=50, input_shape=(
        X_train.shape[1],
        X_train.shape[2]),
         activation='relu')
         ))


model.add(Dense(1))
opt = Adam()
model.compile(loss='mean_squared_error', optimizer=opt)


# ## Fit Model To Train Dataset

# In[14]:

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, shuffle=True)

# ## Plot Losses Per Epoch

# In[15]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## Make Predictions For Test And Train Dataset

# In[16]:


y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)


# ## Convert Values Back To Regular Values

# In[17]:


# Typically you would not scale y values. However, as the scaled total demand values were used to create both the x and y
# values, the scaling for the prediction and y test now needs to be reversed.
y_pred = td_transformer.inverse_transform(y_pred)
y_test = td_transformer.inverse_transform([y_test])

y_pred_train = td_transformer.inverse_transform(y_pred_train)
y_train = td_transformer.inverse_transform([y_train])


# ## Calculate RMSE

# ### Test Dataset

# In[18]:


y_test = y_test.reshape(-1)
y_test = pd.Series(y_test)

y_pred = y_pred.reshape(-1)
y_pred = pd.Series(y_pred)

rmse = mean_squared_error(y_test, y_pred, squared = False)

print('Test RMSE: %f' % rmse)


# ### Train Dataset (for comparison with Test dataset)

# In[19]:


y_train = y_train.reshape(-1)
y_train = pd.Series(y_train)

y_pred_train = y_pred_train.reshape(-1)
y_pred_train = pd.Series(y_pred_train)

rmse = mean_squared_error(y_train, y_pred_train, squared = False)

print('Train RMSE: %f' % rmse)


# ## Average Percentage Difference Between Prediction And Actual Values

# In[20]:


predict_actual = y_test.to_frame()
predict_actual['PREDICTION'] = y_pred
predict_actual.columns = ['ACTUAL', 'PREDICTION']
predict_actual['PERCENT_DIFF'] = abs(predict_actual.ACTUAL - predict_actual.PREDICTION) / predict_actual.ACTUAL * 100


# In[21]:


avg_perc_diff = sum(predict_actual.PERCENT_DIFF) / len(predict_actual)
print('Average Percentage Difference', avg_perc_diff)

# ## Save The Result in a .csv with Timestamp, Temperature, Actual & Predicted Demand

# In[22]:


df = df[(df.YEAR == 2021)]

# reset the index as it is a filtered set
df.reset_index(drop=True, inplace=True) 

# drop the records used for the first prediction
df.drop(range(0,time_steps), axis = 0, inplace=True) 

# reset the index as we dropped records.
df.reset_index(drop=True, inplace=True) 

print(predict_actual.head())
print(df.head())
KeyData = ['DATETIME', 'TEMPERATURE', 'TOTALDEMAND']

df_out = pd.concat([df[KeyData], predict_actual['PREDICTION']], axis=1)
df_out['Percentage diff'] = 100 * (df_out['PREDICTION'] - df_out['TOTALDEMAND']) / df_out['TOTALDEMAND']
print(df_out.head())
df_out.to_csv('./output/LSTM_result.csv', index = False)

# ## Save The Model AND Scalers
# In[23]:

# ### Save The Modle
model.save('./output/LSTM_model.h5')  # creates a HDF5 file

# ### Save The Scalers For Demand AND Temporarture
dump(td_transformer, open('./output/scaler_td.pkl', 'wb'))
dump(temp_transformer, open('./output/scaler_temperature.pkl', 'wb'))


