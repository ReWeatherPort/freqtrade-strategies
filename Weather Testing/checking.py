

import ccxt

# Replace with your Binance API key and secret
api_key = 'LC0zOTGpwxL1Z0lawKGnQNst1adjg6o2ER5hLQsX4DAABRIIiDMZNLctdxA9vDsM'
api_secret = 'ycHOwRbsws0CVZmDKQHBHhG2CrYNK4UDiPw7lKkFTaPBWAKlBQttk1upADSmQLaq'

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {
        'adjustForTimeDifference': True,  # Automatically sync with Binance's server time
        'useServerTime': True             # Use Binance's server time
    }
})

try:
    # Fetch account balance to test API key
    balance = exchange.fetch_balance()
    print("API Key is valid! Balance fetched successfully.")
except ccxt.AuthenticationError as e:
    print("Invalid API Key or Secret:", e)
except Exception as e:
    print("Other error:", e)