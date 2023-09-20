import os
import symbol
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


# Load the companies to symbols mapping from a pickled file
with open('symbols_to_companies.pkl', 'rb') as f:
    companies_to_symbols = pickle.load(f)


# Function to calculate Cumulative Abnormal Returns (CAR)
def calculate_car(stock_data, market_data, event_date, window_before, window_after):
    print(f"Calculating CAR for event_date: {event_date}")

    # Calculate daily returns for the stock and the market
    stock_returns = stock_data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()

    # Define the reference window around the event date
    reference_window_start = event_date - pd.DateOffset(315)
    reference_window_end = event_date - pd.DateOffset(15)

    # Check for sufficient data in the reference window
    if reference_window_start < stock_data.index[0] or reference_window_end > stock_data.index[-1]:
        print(f'Not enough data in the reference window for event on {event_date}')
        return np.nan

    # Fit a linear regression model to estimate the expected returns
    model = LinearRegression()
    model.fit(market_returns.loc[reference_window_start:reference_window_end].values.reshape(-1, 1),
              stock_returns.loc[reference_window_start:reference_window_end].values.reshape(-1, 1))

    # Use the model to predict the expected returns during the event window
    event_window_start = event_date - pd.DateOffset(window_before)
    event_window_end = event_date + pd.DateOffset(window_after)

    # Check if the company's market cap is > $2 billion during the event window
    # Define market cap condition for the specific event window range
    market_cap_condition = (stock_data.loc[event_window_start:event_window_end, 'MarketCap'] >= 2000000000)

    # Apply the condition to the event window range
    within_range_events = stock_data.loc[event_window_start:event_window_end][market_cap_condition]

    within_range_events['Symbol'] = symbol

    # Save events with market cap less than $2 billion to a CSV file
    if not within_range_events.empty:
        # Check if the file exists and is empty
        if not os.path.isfile('MarketCapExclusionEvent.csv') or os.stat('MarketCapExclusionEvent.csv').st_size == 0:
            # If the file is empty, write data with headers
            within_range_events.to_csv('MarketCapExclusionEvent.csv', mode='a')
        else:
            # If the file is not empty, write data without headers
            within_range_events.to_csv('MarketCapExclusionEvent.csv', mode='a', header=False)

    # Predict the expected returns for the event window
    expected_returns = model.predict(market_returns.loc[event_window_start:event_window_end].values.reshape(-1, 1))

    # Calculate abnormal returns as the difference between actual returns and expected returns
    abnormal_returns = stock_returns.loc[event_window_start:event_window_end] - expected_returns.ravel()

    # Calculate CAR as the sum of the abnormal returns
    car = abnormal_returns.sum()
    return car


# Function to perform CAR calculations for all companies and multiple event windows
def calculate_all_cars():

    # Load symbols to companies mapping, market data, and FDA data
    with open('symbols_to_companies.pkl', 'rb') as f:
        symbols_to_companies = pickle.load(f)

    # Load market data
    market_data = pd.read_csv('market_data.csv', index_col='Date', parse_dates=True)

    # Load FDA data
    fda_data = pd.read_csv('fda_data.csv')

    # Define different event windows for CAR calculation
    windows = [(15, 15), (10, 10), (5, 5), (0, 10), (5, 15), (0, 15), (5, 30), (0, 30)]

    # Initialize DataFrame to store the CAR results
    results = pd.DataFrame(columns=['Company', 'Event Date', 'Window', 'CAR', 'Posted Citations', 'Classification',
                                    'Project Area', 'Product Type'])

    # Initialize DataFrame to hold excluded events
    excluded_events = pd.DataFrame(columns=['Symbol', 'Date', 'MarketCap'])

    # Loop through each company and calculate CAR for its events
    for company, symbol in symbols_to_companies.items():
        print(f"Processing ticker: {symbol} for company: {company}")

        # Check for the presence of stock data file
        filename = f'{symbol}_data.csv'
        if not os.path.isfile(filename):
            print(f"Stock data file for {symbol} does not exist. Skipping calculations.")
            continue

        # Load the stock data
        stock_data = pd.read_csv(filename, parse_dates=True, index_col='Date')

        # Get FDA events for company
        company_fda_data = fda_data[fda_data['Company Name'] == company]

        # Loop through each FDA event and calculate CAR
        for event in company_fda_data.itertuples():
            event_date_str = event[10]  # Get the event date as a string
            posted_citations = event[9]  # Get the Posted Citations
            classification = event[11]  # Get the Classification
            project_area = event[12]  # Get the Project Area
            product_type = event[13]  # Get the Product Type
            try:
                event_date = pd.to_datetime(event_date_str)  # Convert the event date to datetime format
                if event_date not in stock_data.index:
                    continue  # Skip this event if there is no stock data for the event date

                # Calculate CAR for each event window
                for window_before, window_after in windows:
                    try:
                        car = calculate_car(stock_data, market_data, event_date, window_before, window_after)
                        results = results.append({'Company': company,
                                                  'Event Date': event_date,
                                                  'Window': f'{window_before}_{window_after}',
                                                  'CAR': round(car, 11),  # Round CAR to 11 decimal places
                                                  'Posted Citations': posted_citations,
                                                  'Classification': classification,
                                                  'Project Area': project_area,
                                                  'Product Type': product_type}, ignore_index=True)
                    except Exception as e:
                        print(f'Error calculating CAR for {company} on {event_date}: {e}')
            except ValueError:
                print(f'Error processing date for {company}: Unknown string format: {event_date_str}')

            # Save excluded events to CSV file
            excluded_events.to_csv('MarketCapExclusionEvent.csv')

            # Save results to a CSV file
            results.to_csv('car_results.csv', index=False)

            # Remove any NaN values from the results
            results = results.dropna(how='any')

            # Save the cleaned results to a new CSV file
            results.to_csv('car_results_cleaned.csv', index=False)


# Main entry point for the script
if __name__ == '__main__':
    calculate_all_cars()
