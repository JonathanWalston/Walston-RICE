import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binom_test, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations


def calculate_stats():

    # Function to compute various statistical tests on CAR for a given windowz and categoryz
    def calculate_tests(df, windowz, categoryz=None, category_valuez=None):
        # Filter data based on categoryz and value if provided
        if categoryz and category_valuez:
            df = df[df[categoryz] == category_valuez]
        window_data = df[df['Window'] == windowz]['CAR']

        # Compute descriptive statistics for the CAR data
        mean = window_data.mean()
        median = window_data.median()
        std = window_data.std()
        min_value = window_data.min()
        max_value = window_data.max()
        sample_size = len(window_data)

        # Perform the Wilcoxon signed-rank test
        if len(window_data) > 0:  # and symmetry_value < 0.01:
            zero_diff = np.zeros(len(window_data))
            wilcoxon_stat, wilcoxon_pz = wilcoxon(window_data, zero_diff)
        else:
            wilcoxon_pz = None

        # Conduct the Binomial test
        binomial_pz = binom_test((window_data > 0).sum(), len(window_data)) if len(window_data) > 0 else None

        # Execute the Mann-Whitney U test
        positive_data = window_data[window_data > 0]
        negative_data = window_data[window_data <= 0]
        if len(positive_data) > 0 and len(negative_data) > 0:
            mannwhitneyu_result = mannwhitneyu(positive_data, negative_data, alternative='two-sided')
            mannwhitneyu_pz = mannwhitneyu_result.pvalue
        else:
            mannwhitneyu_pz = None

        return [windowz, mean, median, std, min_value, max_value, wilcoxon_pz, binomial_pz, mannwhitneyu_pz,
                sample_size, 'VariableX', categoryz, category_valuez]

    # Function to perform Mann-Whitney U test to compare CAR between two windows
    def calculate_mannwhitneyu_between_windows(df, window1, window2, categoryz=None, category_valuez=None):
        # Filter data based on categoryz and value if provided
        if categoryz and category_valuez:
            df = df[df[categoryz] == category_valuez]
        window1_data = df[df['Window'] == window1]['CAR']
        window2_data = df[df['Window'] == window2]['CAR']

        # Conduct the Mann-Whitney U test between the two windows
        if len(window1_data) > 0 and len(window2_data) > 0:
            mannwhitneyu_result = mannwhitneyu(window1_data, window2_data, alternative='two-sided')
            mannwhitneyu_pz = mannwhitneyu_result.pvalue
        else:
            mannwhitneyu_pz = None

        return [window1, window2, mannwhitneyu_pz, categoryz, category_valuez]

    # Load the cleaned CAR results data
    car_results_df = pd.read_csv('car_results_cleaned.csv')

    # Clean the CAR column if it contains any string values, replace leading quotes
    if car_results_df['CAR'].apply(lambda xz: isinstance(xz, str)).any():
        car_results_df['CAR'] = car_results_df['CAR'].str.replace("'", "").astype(float)

    # List of windows for CAR analysis
    windows = ['15_15', '10_10', '5_5', '0_10', '5_15', '0_15', '5_30', '0_30']

    # List of categories to consider in the analysis
    categories = ['Classification', 'Posted Citations', 'Project Area', 'Product Type']

    # Initialize a list to store the results
    results = []

    # Compute statistical tests for each windowz and categoryz
    for window in windows:
        results.append(calculate_tests(car_results_df, window))  # Updated to use car_results_df
        for category in categories:
            for category_value in car_results_df[category].unique():  # Updated to use car_results_df
                results.append(calculate_tests(car_results_df, window, category, category_value))
                # Updated to use car_results_df

    # Compute Mann-Whitney U test between each pair of windows
    pairs = list(combinations(windows, 2))
    pair_results = []

    for pair in pairs:
        pair_results.append(calculate_mannwhitneyu_between_windows(car_results_df, pair[0], pair[1]))
        # Updated to use car_results_df
        for category in categories:
            for category_value in car_results_df[category].unique():  # Updated to use car_results_df
                pair_results.append(calculate_mannwhitneyu_between_windows(car_results_df, pair[0], pair[1], category,
                                                                           category_value))

    # Extract p-values for multiple testing correction
    p_values = []
    for result in results:
        wilcoxon_p = result[6]
        binomial_p = result[7]
        mannwhitneyu_p = result[8]

        if isinstance(wilcoxon_p, (float, int)) and 0 <= wilcoxon_p <= 1:
            p_values.append(wilcoxon_p)
        if isinstance(binomial_p, (float, int)) and 0 <= binomial_p <= 1:
            p_values.append(binomial_p)
        if isinstance(mannwhitneyu_p, (float, int)) and 0 <= mannwhitneyu_p <= 1:
            p_values.append(mannwhitneyu_p)

    for result in pair_results:
        mannwhitneyu_p = result[2]
        if isinstance(mannwhitneyu_p, (float, int)) and 0 <= mannwhitneyu_p <= 1:
            p_values.append(mannwhitneyu_p)

    # Remove any non-numeric elements from p_values
    p_values = [p for p in p_values if isinstance(p, (float, int))]

    # Correct p-values for multiple testing using Benjamini-Hochberg procedure
    p_values = [p for p in p_values if isinstance(p, (float, int)) and 0 <= p <= 1]
    _, corrected_p_values = fdrcorrection(p_values)

    # Create list of indices of valid p-values (only once)
    valid_p_value_indices = [i for i, p in enumerate(p_values) if isinstance(p, (float, int)) and 0 <= p <= 1]

    # Replace old p-values with corrected ones using valid_p_value_indices
    for i, x in enumerate(results):
        if i in valid_p_value_indices:
            x[-3] = corrected_p_values[valid_p_value_indices.index(i)]
    for i, x in enumerate(pair_results):
        if (i + len(results)) in valid_p_value_indices:
            x[-1] = corrected_p_values[valid_p_value_indices.index(i + len(results))]

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Window', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Wilcoxon P-value',
                                                'Binomial P-value', 'Mann-Whitney U P-value', 'Sample Size',
                                                'VariableX', 'Category', 'Category Value'])

    # Convert 'Symmetry Value' to Excel formula
    results_df['Symmetry Value'] = results_df.apply(lambda row: f'=ABS(B{row.name+2}-C{row.name+2})', axis=1)

    results_df.to_csv('car_statistical_results.csv', index=False, float_format='%.10f')


if __name__ == '__main__':
    calculate_stats()
