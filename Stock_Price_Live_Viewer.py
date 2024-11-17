import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Stock Price Viewer"

# App layout
app.layout = html.Div([
    html.H1("Stock Price Viewer", style={'text-align': 'center'}),

    # Input for ticker symbol
    html.Div([
        html.Label("Enter Ticker Symbol:", style={'font-weight': 'bold'}),
        dcc.Input(id='ticker-input', type='text', value='AAPL', placeholder="Enter stock ticker..."),
    ], style={'margin-bottom': '20px'}),

    # Dropdown for selecting time period
    html.Div([
        html.Label("Select Time Period:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='time-period',
            options=[
                {'label': '1 Day', 'value': '1d'},
                {'label': '5 Days', 'value': '5d'},
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label': '6 Months', 'value': '6mo'},
                {'label': '1 Year', 'value': '1y'},
                {'label': '2 Years', 'value': '2y'},
                {'label': '5 Years', 'value': '5y'},
                {'label': 'Year-to-Date', 'value': 'ytd'},
                {'label': 'Max', 'value': 'max'}
            ],
            value='1mo',  # Default value
            clearable=False
        ),
    ], style={'margin-bottom': '20px'}),

    # Div for displaying the current stock price
    html.Div(id='current-price', style={'font-size': '20px', 'margin-bottom': '20px', 'text-align': 'center'}),

    # Graph for displaying stock prices
    dcc.Graph(id='stock-chart'),

    # Interval for automatic updates (30 seconds)
    dcc.Interval(
        id='interval-component',
        interval=30 * 1000,  # 30 seconds in milliseconds
        n_intervals=0
    ),

    # Div for displaying error messages
    html.Div(id='error-message', style={'color': 'red', 'margin-top': '20px'})
])

# Callback to update the graph and current price based on user inputs and interval updates
@app.callback(
    [Output('stock-chart', 'figure'),
     Output('current-price', 'children'),
     Output('error-message', 'children')],
    [Input('ticker-input', 'value'),
     Input('time-period', 'value'),
     Input('interval-component', 'n_intervals')]  # Triggered by interval
)
def update_chart(ticker, time_period, n_intervals):
    if not ticker:
        return {}, "Please enter a valid ticker symbol.", ""

    try:
        # Fetch stock data
        stock_data = yf.download(ticker, period=time_period, interval='1h')  # 1-hour interval
        if stock_data.empty:
            return {}, "", f"No data found for ticker symbol: {ticker}"

        # Fetch current stock price
        ticker_obj = yf.Ticker(ticker)
        history_data = ticker_obj.history(period='1d')
        if history_data.empty:
            return {}, "", f"Unable to fetch current price for {ticker}. No recent trading data available."
        
        current_price = history_data['Close'][-1]

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
            title=f"{ticker.upper()} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        # Current price message
        current_price_message = f"Current Price for {ticker.upper()}: ${current_price:.2f} (Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"

        return fig, current_price_message, ""
    except Exception as e:
        return {}, "", f"Error fetching data: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
