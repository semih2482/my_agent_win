import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

def run_granger_causality(data: pd.DataFrame, response_var: str, predictor_var: str, max_lag: int = 4) -> dict:
    """
    Performs a Granger causality test to see if one time series is useful in forecasting another.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the time series data.
                             The data should be stationary for the test to be meaningful.
        response_var (str): The column name of the time series that might be predicted (the response).
        predictor_var (str): The column name of the time series that might be a predictor.
        max_lag (int): The maximum number of lags to test for causality.

    Returns:
        dict: A dictionary containing the test results for each lag. 
              The keys are the lags, and the values are the p-values of the F-test.
              A low p-value (< 0.05) suggests that the predictor variable Granger-causes the response variable.
              Returns an error message if the test fails.
    """
    try:
        # Select and clean the data
        causality_data = data[[response_var, predictor_var]].dropna()

        if len(causality_data) < 3 * max_lag:
            return {"error": f"Not enough data to perform Granger causality test with max_lag={max_lag}."}

        print(f"Running Granger causality test: Does '{predictor_var}' Granger-cause '{response_var}'?")

        # The test requires a numpy array
        test_result = grangercausalitytests(causality_data[[response_var, predictor_var]], maxlag=max_lag, verbose=False)

        results = {}
        for lag in range(1, max_lag + 1):
            f_test_p_value = test_result[lag][0]['ssr_ftest'][1]
            results[f'lag_{lag}'] = f_test_p_value
        
        print("Granger causality test completed.")
        return results

    except Exception as e:
        print(f"An error occurred during Granger causality test: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # Create a sample DataFrame for demonstration
    # In a real scenario, this data would be fetched and pre-processed (e.g., for stationarity)
    np.random.seed(42)
    n_obs = 100
    dates = pd.to_datetime(pd.date_range('2022-01-01', periods=n_obs, freq='D'))
    
    # A series that is influenced by the past of another series
    predictor = np.random.randn(n_obs).cumsum()
    response = 0.5 * np.roll(predictor, 1) + np.random.randn(n_obs).cumsum()
    
    df = pd.DataFrame({
        'Date': dates,
        'Stock_Returns': response,
        'Oil_Price_Changes': predictor
    }).set_index('Date')

    # Test 1: Check if Oil_Price_Changes Granger-cause Stock_Returns
    print("--- Test Case 1 ---")
    causality_results = run_granger_causality(df, 'Stock_Returns', 'Oil_Price_Changes', max_lag=3)

    if 'error' in causality_results:
        print(f"Error: {causality_results['error']}")
    else:
        print("P-values for each lag:")
        for lag, p_value in causality_results.items():
            print(f"  {lag}: {p_value:.4f}")
        # A low p-value at a certain lag suggests a causal relationship at that lag.

    # Test 2: Check the reverse relationship
    print("\n--- Test Case 2 ---")
    reverse_causality_results = run_granger_causality(df, 'Oil_Price_Changes', 'Stock_Returns', max_lag=3)
    if 'error' in reverse_causality_results:
        print(f"Error: {reverse_causality_results['error']}")
    else:
        print("P-values for each lag (reverse direction):")
        for lag, p_value in reverse_causality_results.items():
            print(f"  {lag}: {p_value:.4f}")
