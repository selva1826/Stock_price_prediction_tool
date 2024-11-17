from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from threading import Thread
import requests

# Flask app
server = Flask(__name__)

@server.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker')
    timeline = data.get('timeline')

    if not ticker or not timeline:
        return jsonify({'error': 'Missing ticker or timeline'}), 400

    try:
        # Fetch historical data
        stock_data = yf.download(ticker, period='5y', interval='1d')
        if stock_data.empty:
            return jsonify({'error': f"No data found for ticker: {ticker}"}), 404

        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        train_data = scaled_data[:int(len(scaled_data) * 0.8)]
        X_train, y_train = [], []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build and train LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        # Predict future prices
        prediction_days = {'1mo': 30, '3mo': 90, '6mo': 180}
        last_60_days = scaled_data[-60:]
        X_test = last_60_days.reshape(1, -1, 1)

        predictions = []
        for _ in range(prediction_days[timeline]):
            pred = model.predict(X_test)
            predictions.append(pred[0][0])
            X_test = np.append(X_test[:, 1:], pred).reshape(1, -1, 1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        prediction_dates = pd.date_range(start=stock_data.index[-1], periods=len(predicted_prices) + 1).strftime('%Y-%m-%d')[1:]

        # Combine historical and predicted data for the chart
        historical_prices = stock_data['Close'][-120:]  # Last 120 days of historical data
        historical_dates = stock_data.index[-120:].strftime('%Y-%m-%d')

        return jsonify({
            'historical_dates': list(historical_dates),
            'historical_prices': historical_prices.tolist(),
            'prediction_dates': list(prediction_dates),
            'predicted_prices': predicted_prices.flatten().tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Dash app embedded in Flask
dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

dash_app.layout = html.Div([
    html.H1("Stock Price Viewer & Predictor", style={'text-align': 'center'}),
    html.Div([
        html.Label("Enter Ticker Symbol:", style={'font-weight': 'bold'}),
        dcc.Input(id='ticker-input', type='text', value='AAPL', placeholder="Enter stock ticker..."),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Label("Select Prediction Timeline:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='prediction-timeline',
            options=[
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label': '6 Months', 'value': '6mo'},
            ],
            value='1mo',
            clearable=False
        ),
    ], style={'margin-bottom': '20px'}),
    dcc.Graph(id='stock-chart', style={'margin-bottom': '40px'}),
    html.Div([
        html.H3("Stock Price Prediction", style={'text-align': 'center'}),
        dcc.Graph(id='prediction-chart'),
        html.Div(id='prediction-error', style={'color': 'red', 'text-align': 'center'})
    ])
])


@dash_app.callback(
    [Output('stock-chart', 'figure'),
     Output('prediction-chart', 'figure'),
     Output('prediction-error', 'children')],
    [Input('ticker-input', 'value'),
     Input('prediction-timeline', 'value')]
)
def update_dashboard(ticker, timeline):
    if not ticker:
        return {}, {}, "Please enter a valid ticker symbol."

    try:
        # Fetch current stock data
        stock_data = yf.download(ticker, period='1y', interval='1d')
        if stock_data.empty:
            return {}, {}, f"No data found for ticker symbol: {ticker}"

        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=ticker
            )
        ])
        fig.update_layout(
            title=f"{ticker} Stock Price (Last 1 Year)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        # Call the backend prediction model
        response = requests.post('http://127.0.0.1:5000/predict', json={'ticker': ticker, 'timeline': timeline})
        if response.status_code == 200:
            prediction_data = response.json()
            prediction_fig = go.Figure(data=[
                go.Scatter(
                    x=prediction_data['prediction_dates'],
                    y=prediction_data['predicted_prices'],
                    mode='lines+markers',
                    name='Predicted Prices'
                )
            ])
            prediction_fig.update_layout(
                title=f"Predicted Stock Price for {ticker} ({timeline})",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )
            return fig, prediction_fig, ""
        else:
            return fig, {}, "Prediction service error. Please try again."

    except Exception as e:
        return {}, {}, f"Error fetching data: {str(e)}"


# Run Flask and Dash
if __name__ == '__main__':
    Thread(target=lambda: server.run(port=5000)).start()  # Start Flask server
    dash_app.run_server(debug=True, port=8050)            # Start Dash server
