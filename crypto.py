import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

st.title('CRYPTO FORECAST')
crypto_coins = ['BITCOIN','ETHEREUM','DOGECOIN','LITECOIN','BITCOIN CASH']
tics = ['BTC-USD','ETH-USD','DOGE-USD','LTC-USD','BCH-USD']
metrics = ['Open','High','Low','Close']

today = datetime.datetime.today()
today
date_today = today.strftime("%Y-%m-%d")
end_date = date_today
start_date = '2015-01-01'

cry = st.selectbox('Select a crypto coin: ',crypto_coins)
st.write(f'Showing the data upto {str(date_today)[:10]}: ',cry)


@st.cache_data

def get_data(cry):
    for i in range(len(tics)):
        if cry == crypto_coins[i]:
            ticker = tics[i]

    df = yf.download(ticker, start = start_date, end = end_date)
    data = pd.DataFrame(df)
    data['Date'] = pd.to_datetime(data.index).date
    data = data.reset_index(drop=True)

    return data
data = get_data(cry)
st.write(data.tail())

st.write('Data summary until today')
st.write(data.describe())

data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['day'] = pd.to_datetime(data['Date']).dt.day



fig1 = px.scatter(data,x=data['Date'],y=data['Close'],color = 'Year')

st.plotly_chart(fig1)

times = ['day','Month','Year']
days = list(data['day'].unique())
months = list(data['Month'].unique())
years = list(data['Year'].unique())

met = st.multiselect('Select multiple metrics to gain insught: ',metrics)
me_se = list(met)
months_opt = st.multiselect('Select months to gain insight: ',months)
months_opted = list(months_opt)
years_opt = st.multiselect('Select months to gain insight: ',years)
years_opted = list(years_opt)


fig3 = px.scatter(data[(data['Month'].isin(months_opted)) & data['Year'].isin(years_opted)],x='Date',y=me_se)
st.plotly_chart(fig3)



x = data[['Open','High','Low','Close']].values
y = data[['Open','High','Low','Close']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=180)
model = Sequential()
model.add(LSTM(120, activation='relu', return_sequences=True, input_shape=(4, 1)))
model.add(LSTM(120, activation='relu'))
model.add(Dense(80, activation='linear'))
model.add(Dense(4, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])

progress = st.progress(0,text = 'Please wait while model is understanding the data!! ⏳')
for epoch in range(10):
    model.fit(x,y,batch_size = 10, verbose=2)
    progress.progress((epoch+1)*10,text=f'loading {(epoch+1)*10}% ⏳')

st.success('Model trained successfully using data!! ⌛')
status = 'Created model using data.'
st.write(status)
y_pred = model.predict(x_test)
y_pred=y_pred.reshape(y_test.shape)

st.write('Difference between the actual and predicted values of Close')
diff = y_test[:,2:3] - y_pred[:,2:3]
fig4 = px.scatter(diff)
st.plotly_chart(fig4)

import datetime

today = datetime.date.today()

# Create a date input widget
date_input = st.date_input('Select a date:', value=today)
n = (date_input - today).days 
n = int(n)
st.write(f"Selected {n} days for forecast")

last = list(data.iloc[-1,0:4]) 
forecastn = [last] 
for i in range(1,n+1):
  forecastn.append(model.predict([forecastn[i-1]]))

forecastn.pop(0)

def sep(extra):
    for i in range(len(extra)):
        extra[i] = list(extra[i])
    extra1 = [arr[0] for arr in extra]

    return extra1

close_n = [arr[0][3:4] for arr in forecastn]
close_n = sep(close_n)
open_n = [arr[0][0:1] for arr in forecastn]
open_n = sep(open_n)
high_n = [arr[0][1:2] for arr in forecastn]
high_n = sep(high_n)
low_n = [arr[0][2:3] for arr in forecastn]
low_n = sep(low_n)

def n_dates(n):
  today = datetime.date.today()
  dates = []
  for i in range(n):
    dates.append(today + datetime.timedelta(days=i))
    dates[i] = dates[i].strftime("%Y-%m-%d")
  return dates

dates_n = n_dates(n)
forecast_df = pd.DataFrame({'Open':open_n,'High':high_n,'Low':low_n,'Close':close_n,'Date':dates_n})

st.write('Forecast results')
st.table(forecast_df)

existed_df = pd.DataFrame({'Open':data['Open'],'High':data['High'],'Low':data['Low'],'Close':data['Close'],'Date':data['Date']})

metrics = ['Open','High','Low','Close']
metric = st.selectbox('Select a metric: ',metrics)

existed_trace = go.Scatter(x=existed_df['Date'],y=existed_df[metric],mode='markers', name='Existing Data')
forecast_trace = go.Scatter(x=forecast_df['Date'],y=forecast_df[metric],mode='markers', name='forecast Data')
layout = go.Layout(title=f"{metric} Values - existing and forecasted data", xaxis=dict(title='Date'), yaxis=dict(title=f'{metric} Value'))
fig5 = go.Figure(data=[existed_trace, forecast_trace], layout=layout)

st.plotly_chart(fig5)

def impact_arr(arr):
    impact = ['inc']
    for i in range(1,len(arr)):

        if arr[i] - arr[i-1] >0:
            impact.append('inc')
        elif arr[i] - arr[i-1] <0:
            impact.append('dec')
        else:
            impact.append('neu')
    data['impact'] = impact

metric_1 = st.selectbox('Select a metric to analyze its impact: ',metrics)
impact_arr(data[metric_1])
fig6 = px.pie(data,names='impact')
st.write(f'{metric_1} impacts throughtout the history.')
st.plotly_chart(fig6)
