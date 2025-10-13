from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot
 
import pandas as pd


# Read Data
df = pd.read_csv('./mlp_input.csv')

# limit to the recent data.
df = df[(df.year == 2017) | (df.year == 2018) | (df.year == 2019) | (df.year == 2020)]
print(df.head())
size = len(df)

# prepare the categoorical inputs with one hot encoding
df_month = df['month']
df_day   = df['day_name']
df_time  = df['time']

df_month_onehot = pd.get_dummies(df_month)
df_day_onehot   = pd.get_dummies(df_day)
df_time_onehot  = pd.get_dummies(df_time)

# temperature fed as raw value
df_temperature = df['TEMPERATURE']

# prepare the input records
X = pd.concat([df_month_onehot, df_day_onehot, df_time_onehot, df_temperature ], axis=1)

# prepare the forecast 
y = df['TOTALDEMAND']

print(X.head())
print(y.head())

# convert to array to fit the model
X = X.to_numpy()
y = y.to_numpy()


# make the training and test split 70:30
n_train = int(0.70 * size)

trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# Define model
model = Sequential()

# Add hidden layers - Note: dropout didn't help
model.add(Dense(200, input_dim=68, use_bias=True, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='relu'))

# Add loss function and optimiser - adjusting learning rate from default of 0.001 didn't help
#adam = Adam(learning_rate = 0.001)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Fit model with 20% used for validation
history = model.fit(trainX, trainy, validation_split=0.2, epochs=50, batch_size = 48, verbose=0)

# Evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

y_predict = model.predict(X, verbose=0)
rmse = mean_squared_error(y_predict, y, squared = False)
print('total RMSE: ', rmse)

y_predict_test = model.predict(testX, verbose=0)
rmse = mean_squared_error(y_predict_test, testy, squared = False)
print('test RMSE: ', rmse)


# Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('nodp.png')


df_actual   = pd.DataFrame(y, columns = ['Actual Demand'])
df_forecast =  pd.DataFrame(y_predict, columns = ['Forecast Demand'])

# because we have filtered by year, we need to reset index before concat.
df.reset_index(drop=True, inplace=True)
df_out = pd.concat([df, df_actual, df_forecast], axis=1)
df_out['Percentage diff'] = 100 * (df_out['Forecast Demand'] - df_out['Actual Demand']) / df_out['Actual Demand']


print(df_out.head())
df_out.to_csv('mpl_result.csv', index = False)