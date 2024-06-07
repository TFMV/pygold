import json
import os
import numpy as np
from sqlalchemy import create_engine, Column, Float, Date, MetaData, Table
import yaml

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def preprocess_gold_prices(data):
    # Convert to numpy array and handle missing values
    dates = [entry['Date'] for entry in data]
    prices = [entry['Close'] for entry in data if entry['Close'] is not None]
    return np.array(dates), np.array(prices)

def save_to_database(engine, table_name, data):
    metadata = MetaData()
    table = Table(table_name, metadata,
                  Column('date', Date),
                  Column('price', Float))
    metadata.create_all(engine)
    conn = engine.connect()
    conn.execute(table.insert(), data)
    conn.close()

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    engine = create_engine(f"sqlite:///{config['database_path']}")

    gold_prices_data = load_data('data/raw/gold_prices.json')

    dates, prices = preprocess_gold_prices(gold_prices_data)

    save_to_database(engine, 'gold_prices', [{'date': date, 'price': price} for date, price in zip(dates, prices)])

if __name__ == "__main__":
    main()
