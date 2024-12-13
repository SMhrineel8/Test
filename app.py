# Replace fbprophet with statsmodels
import streamlit as st
from datetime import date
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting using Exponential Smoothing
df_train = data['Close']
model = ExponentialSmoothing(df_train, trend='add', seasonal='add', seasonal_periods=12).fit()
forecast = model.forecast(steps=period)

st.subheader('Forecast data')
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=data['Date'].iloc[-1], periods=period+1)[1:],
    'Forecast': forecast
})
st.write(forecast_df.tail())

# Plot forecast
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical'))
fig1.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast'))
fig1.layout.update(title_text='Stock Price Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
