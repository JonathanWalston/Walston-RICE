import pandas as pd
import pickle
import string
from fuzzywuzzy import process
from unicodedata import normalize
from datetime import datetime
import yfinance as yf


# Function to download stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}")
        return None


# Function to preprocess company names by converting to lower-case and removing punctuation
def preprocess_name(name):
    # Remove non-ASCII characters
    name = normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')

    # Convert to lower-case and remove punctuation
    return name.lower().translate(str.maketrans('', '', string.punctuation))


# Function to collect FDA inspection data
def get_fda_data():
    try:
        # Load FDA inspection data from Excel file
        df_inspection = pd.read_excel('FDA Inspection Dataset 11-22.xlsx')

        # Preprocess 'Company Name' column
        df_inspection['Company Name'] = df_inspection['Company Name'].apply(preprocess_name)

        # Filter rows based on 'Classification' column
        df_inspection = df_inspection[df_inspection['Classification'].isin(['Voluntary Action Indicated (VAI)',
                                                                            'Official Action Indicated (OAI)'])]

        # Convert 'Inspection End Date' to datetime format and filter by date range
        df_inspection['Inspection End Date'] = pd.to_datetime(df_inspection['Inspection End Date'])
        start_date = datetime(2009, 1, 1)
        end_date = datetime(2019, 12, 31)
        df_inspection = df_inspection[
            (df_inspection['Inspection End Date'] >= start_date) & (df_inspection['Inspection End Date'] <= end_date)]

        # Select required columns
        df_inspection = df_inspection[
            ['FEI Number', 'Company Name', 'City', 'State', 'Zip', 'Country/Area', 'Fiscal Year',
             'Inspection ID', 'Posted Citations', 'Inspection End Date', 'Classification',
             'Project Area', 'Product Type', 'FMD-145 Date']]
        # df_inspection.dropna(inplace=True)
    except Exception as e:
        print(f"An error occurred while collecting FDA data: {str(e)}")
        return None
    return df_inspection


# Function to collect company data including legal names and ticker symbols
def get_company_data():
    try:
        # Load legal names and ticker symbols from Excel file
        df_legal = pd.read_excel('export_05222023101206.xlsx')

        # Preprocess 'Legal Name' column
        df_legal['Legal Name'] = df_legal['Legal Name'].apply(preprocess_name)

        # Filter rows based on 'EXCHANGE NAME' column
        df_legal = df_legal[df_legal['EXCHANGE NAME'].isin(['NEW YORK STOCK EXCHANGE', 'NASDAQ', 'NYSE'])]

        # Select required columns and drop NaN rows
        df_legal = df_legal[['DUNS NUMBER', 'Legal Name', 'COUNTRY NAME', 'TICKER SYMBOL', 'EXCHANGE NAME']]
        df_legal.dropna(inplace=True)

    except Exception as e:
        print(f"An error occurred while collecting company data: {str(e)}")
        return None

    return df_legal


# Function to load predicted matches from CSV file
def get_predicted_matches():
    try:
        df_predicted = pd.read_csv('predicted_output_updated.csv')
        df_predicted = df_predicted[df_predicted['Predicted Match'] == 1]
    except Exception as e:
        print(f"An error occurred while loading predicted matches: {str(e)}")
        return None

    return df_predicted


def get_master_list():
    try:
        df_master = pd.read_csv('master_list.csv')
        df_master['Subsidiaries'] = df_master['Subsidiaries'].apply(preprocess_name)
    except Exception as e:
        print(f"An error occurred while loading master list: {str(e)}")
        return None
    return df_master


# Main function to orchestrate data collection and processing
def collect_data():
    df_fda = get_fda_data()
    df_export = get_company_data()
    df_predicted = get_predicted_matches()
    df_master = get_master_list()

    # Check for errors in data collection
    if df_fda is None or df_export is None or df_predicted is None or df_master is None:
        print("Error in data collection. Exiting.")
        return

    # Dictionary to map company names to their stock symbols
    symbols_to_companies = {}
    companies_included = []
    companies_excluded = []

    # Match FDA companies with stock symbols
    for _, row in df_fda.iterrows():
        company_name = preprocess_name(row['Company Name'])
        predicted_legal_name_row = df_predicted[df_predicted['Company Name'] == company_name]

        if not predicted_legal_name_row.empty:
            predicted_legal_name = preprocess_name(predicted_legal_name_row['Predicted Legal Name'].values[0])
            exact_match_row = df_export[df_export['Legal Name'] == predicted_legal_name]

            if not exact_match_row.empty:
                symbol = exact_match_row['TICKER SYMBOL'].values[0]
                symbols_to_companies[company_name] = symbol

            else:
                best_match_score = process.extractOne(company_name, df_master['Subsidiaries'])
                if best_match_score:

                    if len(best_match_score) >= 2:
                        best_match, score = best_match_score[:2]

                        if score > 85:
                            match_in_master = df_master[df_master['Subsidiaries'] == best_match]
                            if not match_in_master.empty:
                                symbol = match_in_master['Ticker'].values[0]
                                symbols_to_companies[company_name] = symbol
                    else:
                        print(f"Unexpected length of tuple for {company_name}")
                else:
                    print(f"No match found for {company_name}")

    # Download stock data and filter companies based on market capitalization
    for company_name, symbol in symbols_to_companies.items():
        try:
            data = get_stock_data(symbol, '2008-01-01', '2020-12-31')

            if data is not None:
                ticker = yf.Ticker(symbol)
                info = ticker.get_info()
                shares_outstanding = info.get('sharesOutstanding', 1)  # use 1 as a fallback
                data['MarketCap'] = data['Close'] * shares_outstanding
                data['Shares'] = shares_outstanding

                if data['MarketCap'].mean() > 2000000000:
                    data.to_csv(f'{symbol}_data.csv')
                    companies_included.append((symbol, company_name))
                else:
                    companies_excluded.append((symbol, company_name))
        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")

    # Save the symbols_to_companies dictionary
    with open('symbols_to_companies.pkl', 'wb') as f:
        pickle.dump(symbols_to_companies, f)

    # Save the companies_included and companies_excluded data
    pd.DataFrame(companies_included, columns=['Ticker', 'Company Name']).to_csv('companies_included.csv', index=False)
    pd.DataFrame(companies_excluded, columns=['Ticker', 'Company Name']).to_csv('companies_excluded.csv', index=False)

    # Fetch and save the market data for the S&P 500 index
    market_data = get_stock_data('^GSPC', '2008-01-01', '2020-12-31')
    if market_data is not None:
        market_data.to_csv('market_data.csv')

    # Save the preprocessed FDA data
    df_fda.to_csv('fda_data.csv', index=False)


# Main entry point of the script
if __name__ == "__main__":
    collect_data()
