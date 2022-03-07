
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import mean_absolute_error
import streamlit as st
import klib

st.write("######energy prediction ccpp")

df = pd.read_csv('energy_production (2).csv',sep=";")
st.write("all the variables")
df
klib.data_cleaning(df)
klib.corr_plot(df, split='pos') 
klib.corr_plot(df, split='neg')
klib.dist_plot(df)
klib.corr_mat(df)

df.rename(columns={"temperature":"AT" , "exhaust_vacuum":"V" , "amb_pressure":"AP" , "r_humidity":"RH" , "energy_production": "PE"}, inplace=True)
df_4 = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(df_4, y, test_size = 0.2, random_state = 0)

rf_regressor= RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('rmse is : ', rmse)
st.write(" RMSE is ", rmse)
r_squared = r2_score(y_test, y_pred)
print("r_squared is : ", r_squared)
st.write( R^2 is ", r_squared)
mae = mean_absolute_error(y_test, y_pred)
print('mae is : ', mae)
st.write(" MAE is ", mae)
