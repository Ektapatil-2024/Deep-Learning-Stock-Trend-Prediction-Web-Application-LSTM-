import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
import random
import time



start = '2015-01-01'
end = '2025-01-22'

st.set_page_config(page_title="My Streamlit App", layout="wide")
st.image('Stock Market Predictions.jpg', width=450)
st.title(':rainbow[Stock Trend Prediction]')


ticker_symbol='AAPL'
user_input=st.text_input('Enter Stock Ticker',ticker_symbol)
df = yf.download(user_input, start=start, end=end)

def currentPrice(user_input):
    ticker = yf.Ticker(user_input)
    current_price = ticker.history(period="1d")['Close'][0]
    st.markdown(f"The latest closing price of '{user_input}'  is: :blue[${current_price:.2f}]")

def historicalData(user_input):
    ticker = yf.Ticker(user_input)
    historical_data = ticker.history(period="5d")
    st.subheader('Historical Data (Last 5 Days)')
    st.text(historical_data['Close'])  


# Display the current market price (last closing price)
currentPrice(user_input)

# Display historical data (e.g., last 5 days)
historicalData(user_input)

# Describing Data
df.columns = df.columns.droplevel('Ticker')
df=df.reset_index()
df.head() 
st.subheader('Data from 2015-2025')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Visualization of Closing Price vs Time Chart with 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#Visualization of Closing Price vs Time Chart with 100MA & 200MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r',label= 'MA100')
plt.plot(ma200, 'b',label= 'MA200')
plt.plot(df.Close, 'g')
plt.legend()
st.pyplot(fig)

#Splitting data into Training and Testing
data_training =pd.DataFrame (df['Close'][0:int(len(df)*0.70)])
data_testing= pd.DataFrame (df['Close'][int(len(df)*0.70):int(len(df))])

#Model Creation
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)

#Load model
model = load_model('keras_model.h5')

#Testing
past_100_days= data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data= scaler.fit_transform(final_df)

#Splitting data into x_test and y_test
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test= np.array(x_test),np.array(y_test)

y_predicted= model.predict(x_test)

scaler = scaler.scale_

scale_factor=1/scaler[0]
y_predicted= y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label= 'Orginal Price')
plt.plot(y_predicted,'r',label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# # ===================================================================================================================
# Chatbot
# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! Which stock can I assist you with today?",
            "Hi, human! Which stock can I assist you withtoday?",
            "Which stock do you want to see?"
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Accept user input
with st.sidebar:
     st.title('Stock Trends Bot')
     if prompt := st.chat_input("Hi, What's up?"):
         # Add user message to chat history
         st.session_state.messages.append({"role": "user", "content": prompt})
         with st.chat_message("user"):
             st.markdown(prompt)
         with st.chat_message("assistant"):
             response = st.write_stream(response_generator())  
             # Add assistant response to chat history
             st.session_state.messages.append({"role": "assistant", "content": response})
         with st.chat_message("assistant"):
            #  response = st.write_stream(response_generator())
            #  currentPrice(prompt)   
            #  st.session_state.messages.append({"role": "assistant", "content": currentPrice(prompt)})
             historicalData(prompt)

# # ==============================================================================================================
