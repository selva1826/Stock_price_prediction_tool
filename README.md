# Stock Price Viewer & Predictor

This is a web application built using Flask and Dash that allows users to visualize the historical stock prices and predict future stock prices using a machine learning model. The model uses Long Short-Term Memory (LSTM) for stock price prediction based on historical data.

## Features

- **Stock Price Viewer**: Displays the historical stock prices as a candlestick chart.
- **Stock Price Prediction**: Predicts future stock prices based on selected timelines (1 Month, 3 Months, or 6 Months) using an LSTM model.
- **Real-time Updates**: Updates the stock prices and predictions every time the user interacts with the app.

## Technologies Used

- **Flask**: Used to serve the backend API for fetching stock data and predictions.
- **Dash**: Used for creating the interactive frontend dashboard.
- **yfinance**: For fetching historical stock data.
- **TensorFlow/Keras**: For training and running the LSTM model to predict future stock prices.
- **Plotly**: For plotting the stock price charts.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/stock-price-predictor.git
    cd stock-price-predictor
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    python app.py
    ```

   The Flask API will run on port `5000`, and the Dash app will be available at `http://127.0.0.1:8050/`.

## Usage

1. Open the web app in your browser.
2. Enter a stock ticker symbol (e.g., AAPL for Apple) in the input field.
3. Select the timeline for which you want to predict the stock price (1 Month, 3 Months, or 6 Months).
4. The stock chart and the predicted stock prices will be displayed on the page.

## API Endpoint

The app provides a Flask API endpoint to predict the future stock prices:

- **POST /predict**
  - **Request Body**: 
    ```json
    {
      "ticker": "AAPL",
      "timeline": "1mo"
    }
    ```
  - **Response**: 
    ```json
    {
      "historical_dates": ["2023-01-01", "2023-01-02", ...],
      "historical_prices": [150, 152, ...],
      "prediction_dates": ["2024-01-01", "2024-01-02", ...],
      "predicted_prices": [155, 157, ...]
    }
    ```

## Contributing

Feel free to fork the repository, create a branch, and submit a pull request for improvements, bug fixes, or features.
