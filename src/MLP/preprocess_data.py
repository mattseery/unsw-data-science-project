import pandas as pd
    
# Load Data

# Temperature
df_temperature = pd.read_csv('../../data/temperature_nsw_all_values.csv')
df_temperature = df_temperature.drop_duplicates(subset = 'DATETIME', keep = 'first')
print(df_temperature.head())

# Actual Demand
df_demand = pd.read_csv('../../data/totaldemand_nsw.csv')
df_demand = df_demand.drop('REGIONID', axis = 1)
df_demand = df_demand.drop_duplicates(subset = 'DATETIME', keep = 'first')
print(df_demand.head())

# Merge both dataset on DATETIME
result = pd.merge(df_temperature, df_demand, on="DATETIME")

#
# our input for the model will be month, day of the week, and the timeslot.
# but we are carrying the rest to identified the record completely.
#
DatetimeIndex   = pd.DatetimeIndex(result['DATETIME'], dayfirst=True)
result['year']  = DatetimeIndex.year
result['month'] = DatetimeIndex.month_name()
result['day']   = DatetimeIndex.day
result['day_name'] = DatetimeIndex.day_name()
result['time']  = DatetimeIndex.time


result = result.drop('DATETIME', axis = 1)
print(result.head())

# save this to a file to be loaded by model
result.to_csv('mlp_input.csv', index = False)

