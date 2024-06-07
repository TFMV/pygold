import yfinance as yf
import json
import os
import yaml
import datetime

def fetch_gold_prices(ticker='GC=F', start='2010-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    gold_data = yf.download(ticker, start=start, end=end)
    gold_data.reset_index(inplace=True)
    return gold_data.to_dict(orient='records')

def save_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(data, file)

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    gold_prices_data = fetch_gold_prices()
    save_data(gold_prices_data, 'data/raw/gold_prices.json')

if __name__ == "__main__":
    main()
