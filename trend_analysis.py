import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def analyze_and_predict_trend(file_name, periods=30):
    """
    Analyze historical crypto trends and predict future prices.

    Args:
        file_name (str): CSV file with historical price data.
        periods (int): Number of days to predict into the future.

    Returns:
        None
    """
    try:
        # Load historical data
        df = pd.read_csv(file_name)
        
        # Ensure the Date and Close columns exist
        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError("CSV must contain 'Date' and 'Close' columns")
        
        # Clean the data: remove any non-numeric values in 'Close' column
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])  # Drop rows with NaN values

        # Convert date to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Prepare data for Prophet
        data = df[['Date', 'Close']]
        data.columns = ['ds', 'y']
        
        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(data)
        
        # Create future dates for prediction
        future = model.make_future_dataframe(periods=periods)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Plot the forecast
        fig = model.plot(forecast)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{file_name} Trend Prediction")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    analyze_and_predict_trend("BTC-USD_historical_data.csv", periods=30)
