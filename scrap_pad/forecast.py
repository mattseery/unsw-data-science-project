

#Source: https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import mean_squared_error


from matplotlib import pyplot
 

import pandas as pd


# Read Data

feature_cols = ['month', 'day_name', 'time', 'TEMPERATURE']
df = pd.read_csv('./input.csv')

X = df[feature_cols]
y = df['TOTALDEMAND']


# future prove, increase by 20%
max_y = y.max() * 1.1
y = y/max_y

size = len(df)

df_temperature = df['TEMPERATURE']
max_temperature = df_temperature.max() * 1.1
df_temperature = df_temperature/max_temperature

print('max demand: ', max_y)
print('max temperature: ', max_temperature)
print('number of records: ', size)

df_month = df['month']
df_time = df['time']

df_month_onehot = pd.get_dummies(df_month)
df_time_onehot = pd.get_dummies(df_time)


X = pd.concat([df_month_onehot, df_time_onehot, df_temperature ], axis=1)
print(X.head())
print(y.head())

X = X.to_numpy()
y = y.to_numpy()


n_train = int(0.60 * size)
print(n_train)
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# Define model
model = Sequential()
model.add(Dense(100, input_dim=61, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='MSE', optimizer='adam', metrics=['mse'])


# Fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

# Evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

y_predict = model.predict(testX, verbose=0)


# Prediction Done
print('prediction done')

rmse = mean_squared_error(y_predict, testy, squared = False)
print('raw', rmse)

# Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('nodp.png')

df_actual   = pd.DataFrame(testy, columns = ['Actual Demand'])
df_actual   = df_actual * max_y
df_forecast =  pd.DataFrame(y_predict, columns = ['Forecast Demand'])
df_forecast = df_forecast * max_y                         


df_out = pd.concat([df_actual, df_forecast], axis=1)

rmse = mean_squared_error(df_out['Forecast Demand'], df_out['Actual Demand'], squared = False)
print('multiplied', rmse)

print(df_out.head())
df_out.to_csv('result.csv', index = False)

