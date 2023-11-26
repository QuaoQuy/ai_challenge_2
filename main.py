import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler


start = dt.datetime(2017, 1,1)
end = dt.datetime(2024, 6, 6)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AMZN')


if user_input.lower() == 'yahoo':
    st.warning("Please enter a valid stock ticker other than 'yahoo'.")
else:

    df = yf.download(user_input, start=start, end=end)

    st.subheader('Data from 2012-2023')
    st.write(df.describe())
    st.write(df.tail(20))

    st.subheader('Closing Price')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7): int(len(df))])

    print(data_training.shape)
    print(data_testing.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('modellstm2.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Prediction and Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Orignal Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

df = pd.read_csv('AMZN.csv')

start_date = df['Date'].max()
dates = pd.date_range(start_date, periods=30)

X = df[['Close']].values[-100:]

model = load_model('modellstm2.h5')

for date in dates:
    x = X[-100:].reshape((1, 100, 1))

    yhat = model.predict(x)[0]

    X = np.vstack([X[1:], [yhat]])

predictions = X[-30:]

y_pred_original = predictions


data=y_pred_original
normalizedData = (data-np.min(data))/(np.max(data)-np.min(data))
y_pred_original=normalizedData
y_pred_original = y_pred_original*100
import matplotlib.pyplot as plt

st.subheader('Prediction in next 30 days')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_pred_original, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)




