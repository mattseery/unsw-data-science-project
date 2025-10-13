
import pandas as pd
import numpy as np

    
# Load Data



df_temperature = pd.read_csv('./temperature_nsw.csv')
df_temperature = df_temperature.drop('LOCATION', axis = 1)
df_temperature = df_temperature.drop_duplicates(subset = 'DATETIME', keep = 'first')

df_demand = pd.read_csv('./totaldemand_nsw.csv')
df_demand = df_demand.drop('REGIONID', axis = 1)
df_demand = df_demand.drop_duplicates(subset = 'DATETIME', keep = 'first')

result = pd.merge(df_temperature, df_demand, on="DATETIME")
#df_temperature # 220326
#df_demand # 196513

#result # 195947
#result = result.dropna()
#result # 195947 ==> There is no NA

DatetimeIndex   = pd.DatetimeIndex(result['DATETIME'], dayfirst=True)
result['year']  = DatetimeIndex.year
result['month'] = DatetimeIndex.month_name()
result['day']   = DatetimeIndex.day
result['day_name'] = DatetimeIndex.day_name()
result['time']  = DatetimeIndex.time

#result['hour'] = DatetimeIndex.hour
#result['minute'] = DatetimeIndex.minute
#result['dayofweek'] = DatetimeIndex.dayofweek



result = result.drop('DATETIME', axis = 1)
print(result.head())
result.to_csv('input.csv', index = False)


# now workout the adjustment to align the temperture minutes.