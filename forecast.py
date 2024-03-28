# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:33:36 2024

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_excel('C:/Users/DELL/Downloads/Sample_Superstore.xls')

print(df.head())
print(df.tail)
print(df.columns)

print(df.describe())

print(df.dtypes)

print(df.index)

print(df.sample(n=5))

df.columns

df_clean = pd.DataFrame(data=df,columns=['Order Date', 'Region','Product Name', 'Sales'])

# df_clean.columns

# data = df_clean['Sales']
# index_df = df_clean['Order Date']

# index = np.arange(len(df_clean['Order Date']))

# df_new = pd.DataFrame(data,index=index,columns=['Sales'])

# train_size = int(0.8 * len(data))

# train_data = df_new.iloc[:train_size]
# test_data = df_new.iloc[train_size:]

# model = ExponentialSmoothing(train_data,trend='add', seasonal='add', seasonal_periods=12)
# fit_model = model.fit()

# forecast = fit_model.forecast(steps=len(test_data))

# plt.figure(figsize=(10, 6))
# plt.plot(train_data.index, train_data, label='Train data')
# plt.plot(test_data.index, test_data, label='Test data')
# plt.plot(test_data.index, forecast, label='Forecast', linestyle='--')
# plt.title('Holt-Winters Forecast')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

df_clean

def forecast_sales(df):
    forecasts = []
    for region, product in df.groupby(['Region', 'Product Name']):
        region_product_sales = product.set_index('Order Date')['Sales']
        model = ExponentialSmoothing(region_product_sales, seasonal='add', seasonal_periods=12)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=12)  # Forecast for the next 12 months
        forecast.name = f'{region[0]}_{region[1]}'  # Naming convention for the forecast
        forecasts.append(forecast)
    return pd.concat(forecasts, axis=1)

Date =  pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')

Date.dtype

sales_forecast = forecast_sales(df_clean)

df_clean.describe()











