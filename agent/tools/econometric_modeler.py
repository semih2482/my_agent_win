import pandas as pd
import statsmodels.api as sm
from typing import List

def run_linear_regression(data: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> dict:
    """
    Performs an Ordinary Least Squares (OLS) linear regression.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data for the regression.
                             The data should be aligned by date and cleaned (no NaNs).
        dependent_var (str): The name of the column to be used as the dependent variable (Y).
        independent_vars (List[str]): A list of column names to be used as independent variables (X).

    Returns:
        dict: A dictionary containing the model summary as a string and key results 
              (r_squared, adj_r_squared, coefficients, p_values).
              Returns an error message if the regression fails.
    """
    try:
        # Drop rows with missing values in the relevant columns
        clean_data = data[[dependent_var] + independent_vars].dropna()

        if clean_data.empty:
            return {"error": "Data is empty after handling missing values. Cannot run regression."}

        Y = clean_data[dependent_var]
        X = clean_data[independent_vars]
        
        # Add a constant (intercept) to the independent variables
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        results = {
            "summary": str(model.summary()),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
        }
        
        print("Linear regression completed successfully.")
        return results

    except Exception as e:
        print(f"An error occurred during linear regression: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # Create a sample DataFrame for demonstration
    # In a real scenario, this data would be fetched by other tools
    sample_data = {
        'Date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01',
                               '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01']),
        'Stock_Price': [150, 155, 160, 158, 165, 170, 168, 175, 180, 178],
        'Interest_Rate': [0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.75, 0.80, 0.85],
        'GDP_Growth': [0.5, 0.6, 0.55, 0.7, 0.75, 0.8, 0.78, 0.9, 0.92, 0.91]
    }
    df = pd.DataFrame(sample_data).set_index('Date')

    # Define dependent and independent variables
    dep_var = 'Stock_Price'
    ind_vars = ['Interest_Rate', 'GDP_Growth']

    # Run the regression
    regression_results = run_linear_regression(df, dep_var, ind_vars)

    if 'error' in regression_results:
        print(f"\nError: {regression_results['error']}")
    else:
        print("\n--- OLS Regression Results ---")
        print(regression_results['summary'])
        print("\n--- Key Metrics ---")
        print(f"R-squared: {regression_results['r_squared']:.4f}")
        print(f"Adjusted R-squared: {regression_results['adj_r_squared']:.4f}")
        print("\nCoefficients:")
        for var, coef in regression_results['coefficients'].items():
            print(f"  {var}: {coef:.4f}")
