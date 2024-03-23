#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("weather.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.columns


# In[11]:


df.describe()


# In[15]:


plt.figure(figsize=(10, 6))
plt.hist(df['precipitation'], bins=10, color='skyblue', edgecolor='black')
plt.title('Precipitation Histogram')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



# Assuming 'date' is already in datetime format, set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Define the target variable (precipitation in this case)
target_variable = 'precipitation'

# Create lagged features
for i in range(1, 8):  # Lag features for previous 7 days
    df[f'{target_variable}_lag_{i}'] = df[target_variable].shift(i)

# Drop rows with NaN values resulting from the lag operation
df.dropna(inplace=True)

# Define features (lagged precipitation values)
features = [col for col in df.columns if col.startswith(f'{target_variable}_lag_')]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_variable], test_size=0.2, shuffle=False)

# Train a Random Forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Plot actual vs. predicted precipitation
plt.figure(figsize=(10, 6))
plt.plot(df.index, df[target_variable], label='Actual')
plt.plot(X_train.index, train_predictions, label='Train Predictions')
plt.plot(X_test.index, test_predictions, label='Test Predictions')
plt.title('Precipitation Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.show()


# In[ ]:




