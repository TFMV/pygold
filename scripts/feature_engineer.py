import numpy as np
from sqlalchemy import create_engine, MetaData, Table, select
import yaml

def create_features(dates, prices):
    # Create moving averages and percentage changes
    moving_avg = np.convolve(prices, np.ones(5)/5, mode='valid')
    pct_change = np.diff(prices) / prices[:-1]
    return moving_avg, pct_change

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    engine = create_engine(f"sqlite:///{config['database_path']}")
    conn = engine.connect()
    metadata = MetaData()

    gold_prices_table = Table('gold_prices', metadata, autoload_with=engine)

    gold_prices = conn.execute(select([gold_prices_table])).fetchall()

    dates = [entry['date'] for entry in gold_prices]
    prices = [entry['price'] for entry in gold_prices]

    moving_avg, pct_change = create_features(dates, prices)

    features_table = Table('features', metadata,
                           Column('date', Date),
                           Column('moving_avg', Float),
                           Column('pct_change', Float))
    metadata.create_all(engine)
    conn.execute(features_table.insert(), [{'date': date, 'moving_avg': ma, 'pct_change': pc} for date, ma, pc in zip(dates[4:], moving_avg, pct_change)])
    conn.close()

if __name__ == "__main__":
    main()
