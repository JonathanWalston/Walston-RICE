# Walston-RICE

Walston-RICE is a Python-based project focused on collecting and analyzing data related to FDA inspections, company information, and stock market data, with a particular emphasis on calculating Cumulative Abnormal Returns (CAR) around specified event dates.

## Status

The scripts are currently optimized and tested for Windows. Other platforms have not been tested.

## Features

### main.py
- Orchestrates the data collection, CAR calculation, and statistics calculation processes.
- Handles exceptions and provides feedback on the status of each process.

### data_collection.py
- Downloads stock data from Yahoo Finance using the `yfinance` library.
- Preprocesses company names by converting to lowercase, removing punctuation, and handling non-ASCII characters.
- Collects FDA inspection data from an Excel file, preprocessing and filtering the data based on certain criteria.
- Collects company data including legal names and ticker symbols from an Excel file, again preprocessing and filtering the data.
- Loads predicted matches and a master list of companies from CSV files.
- Orchestrates the collection and preprocessing of the datasets, matching FDA companies with stock symbols, downloading stock data, filtering companies based on market capitalization, and saving the processed data in various CSV files.

### car_calculation.py
- Loads a mapping of companies to symbols from a pickled file.
- Calculates Cumulative Abnormal Returns (CAR) for a range of companies around specified event dates, considering different event windows.
- Utilizes a linear regression model to estimate expected returns during the event window.
- Handles various error scenarios, such as missing stock data files and incorrect date formats.
- Stores CAR results in a DataFrame, which is eventually saved to CSV files for further analysis.

### car_stats.py
- Provides a function to compute the power of a two-sided t-test given certain parameters.
- Performs a suite of statistical tests including the Wilcoxon signed-rank test, Binomial test, and Mann-Whitney U test on the CAR data for different windows and categories.
- Conducts a Mann-Whitney U test to compare CAR between two different windows.
- Orchestrates the batch processing of these tests across various configurations of windows, categories, and pairs of windows.
- Organizes the results of the statistical tests into a DataFrame and exports them to a CSV file for further analysis.

## Requirements

**Libraries:**
- pandas
- numpy
- scipy
- statsmodels
- itertools
- pickle
- string
- fuzzywuzzy
- unicodedata
- datetime
- yfinance
- sklearn
- os
- symbol

## Dataset

- FDA Inspection Dataset (`FDA Inspection Dataset 11-22.xlsx`)
- Company Data (`export_05222023101206.xlsx`)
- Predicted Matches (`predicted_output_updated.csv`)
- Master List (`master_list.csv`)
- Stock Data (downloaded from Yahoo Finance)
- Market Data for the S&P 500 index (downloaded from Yahoo Finance)

## Quickstart

1. Clone or download the repository: `git clone https://github.com/JonathanWalston/Walston-RICE.git`
2. Install the required libraries using pip: `pip install pandas numpy scipy statsmodels fuzzywuzzy yfinance sklearn`
3. Open the project in your preferred Python environment.
4. Run `main.py` to execute the data collection, CAR calculation, and statistics calculation processes in sequence.

## Note

The Walston Regulatory Impact Cost Estimator Index (Walston R.I.C.E. Index) offers a unique methodology to quantify the "hidden quality costs" arising from regulatory inspection failures using Cumulative Abnormal Returns (CAR) of companies. 
This index enhances traditional Cost of Quality (COQ) models by introducing a sub-category for regulatory impact costs, helping companies in heavily regulated sectors identify, classify and transition these hidden costs to prevention and appraisal costs. 
The methodology can integrate with conventional accounting models, aiding in budgeting for quality costs based on expected failure costs within modern regulatory frameworks. 
By conducting broad analyses using regulatory inspection datasets across different industries, companies can proactively mitigate losses, showcasing the potential for high return on investment when redirecting such losses towards preventive measures. 
The Walston R.I.C.E. Index advocates for an industry-wide adoption to justify increased prevention costs, thereby reducing failure costs associated with regulatory inspections.

## License

This project is open-source and free to use. Please ensure you follow the terms and conditions laid out in the LICENSE file.
