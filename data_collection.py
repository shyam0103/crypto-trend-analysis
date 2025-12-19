import yfinance as yf

def fetch_crypto_data(symbol, start_date, end_date):
    """
    Fetch historical cryptocurrency data from Yahoo Finance.

    Args:
        symbol (str): Crypto ticker symbol (e.g., 'BTC-USD').
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        None
    """
    try:
        # Fetch data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Reset index to get Date as a column
        data.reset_index(inplace=True)
        
        # Save to CSV (with simple headers)
        file_name = f"{symbol}_historical_data.csv"
        data.to_csv(file_name, index=False, header=True)
        
        print(f"Data saved to {file_name}")
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    fetch_crypto_data("BTC-USD", "2020-01-01", "2024-01-01")
