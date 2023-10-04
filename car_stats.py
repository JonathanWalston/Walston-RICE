import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binom_test, mannwhitneyu
from statsmodels.stats.power import TTestIndPower
from itertools import combinations


# Additional function to calculate power
def calculate_power(n, alpha, effect_size):
    analysis = TTestIndPower()
    return analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0, alternative='two-sided')


def calculate_stats():

    # Function to compute various statistical tests on CAR for a given windowz and categoryz
    def calculate_tests(df, windowz, categoryz=None, category_valuez=None, alphaz=None, effect_sizez=None):
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

        # Inside your calculate_tests function, after each test:
        if wilcoxon_pz is not None:
            wilcoxon_power = calculate_power(sample_size, alphaz, effect_sizez)
        else:
            wilcoxon_power = None

        # Conduct the Binomial test
        binomial_pz = binom_test((window_data > 0).sum(), len(window_data)) if len(window_data) > 0 else None

        # Same for binomial test
        if binomial_pz is not None:
            binomial_power = calculate_power(sample_size, alphaz, effect_sizez)
        else:
            binomial_power = None

        # Execute the Mann-Whitney U test
        positive_data = window_data[window_data > 0]
        negative_data = window_data[window_data <= 0]
        if len(positive_data) > 0 and len(negative_data) > 0:
            mannwhitneyu_result = mannwhitneyu(positive_data, negative_data, alternative='two-sided')
            mannwhitneyu_pz = mannwhitneyu_result.pvalue
        else:
            mannwhitneyu_pz = None

        # And for Mann-Whitney U test
        if mannwhitneyu_pz is not None:
            mannwhitneyu_power = calculate_power(sample_size, alphaz, effect_sizez)
        else:
            mannwhitneyu_power = None

        return [windowz, mean, median, std, min_value, max_value, wilcoxon_pz, wilcoxon_power, binomial_pz,
                binomial_power, mannwhitneyu_pz, mannwhitneyu_power, sample_size, 'VariableX', categoryz,
                category_valuez, alphaz, effect_sizez]

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

    # Define different alphas and effect sizes for sensitivity analysis
    alphas = [0.01, 0.05, 0.10]
    effect_sizes = [0.2, 0.5, 0.8]

    # Initialize a list to store the results
    results = []

    # Compute statistical tests for each windowz and categoryz
    for window in windows:
        for alpha in alphas:
            for effect_size in effect_sizes:
                results.append(calculate_tests(car_results_df, window, alphaz=alpha, effect_sizez=effect_size))
                for category in categories:
                    for category_value in car_results_df[category].unique():
                        results.append(
                            calculate_tests(car_results_df, window, categoryz=category, category_valuez=category_value,
                                            alphaz=alpha, effect_sizez=effect_size))

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

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Window', 'Mean', 'Median', 'Std Dev', 'Min', 'Max',
                                                'Wilcoxon P-value', 'Wilcoxon Power',
                                                'Binomial P-value', 'Binomial Power',
                                                'Mann-Whitney U P-value', 'Mann-Whitney U Power',
                                                'Sample Size', 'VariableX', 'Category', 'Category Value',
                                                'Alpha', 'Effect Size'])

    # Convert 'Symmetry Value' to Excel formula
    results_df['Symmetry Value'] = results_df.apply(lambda row: f'=ABS(B{row.name+2}-C{row.name+2})', axis=1)

    # Update your CSV writing function to include new columns for varying alphas and effect sizes
    results_df.to_csv('car_statistical_results.csv', index=False, float_format='%.10f')


if __name__ == '__main__':
    calculate_stats()
