import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from typing import List, Union

def fetch_macro_data(
    indicators: Union[str, List[str]],
    start_date: str,
    end_date: str,
    source: str = "fred"
) -> pd.DataFrame:
    """
    Fetches macroeconomic data from various online sources using pandas-datareader.

    Args:
        indicators (Union[str, List[str]]): A single indicator code or a list of indicator codes to fetch.
                                            For FRED, examples include 'GDP' (Gross Domestic Product),
                                            'CPIAUCSL' (Consumer Price Index), 'UNRATE' (Unemployment Rate),
                                            'DFF' (Federal Funds Rate).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        source (str): The data source to use. Defaults to 'fred' (Federal Reserve Economic Data).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the requested data, with dates as the index.
                      Returns an empty DataFrame if an error occurs.
    """
    try:
        print(f"Fetching {indicators} from {source} between {start_date} and {end_date}...")
        df = web.DataReader(indicators, source, start_date, end_date)
        print("Data fetched successfully.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage of the function
    # Fetch US GDP and Unemployment Rate from 2020 to 2023
    gdp_unrate_indicators = ['GDP', 'UNRATE']
    start = '2020-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    macro_data = fetch_macro_data(gdp_unrate_indicators, start, end)

    if not macro_data.empty:
        print("\n--- Fetched Macroeconomic Data ---")
        print(macro_data.head())
        print("...")
        print(macro_data.tail())

    # Example for a single indicator
    fed_funds_rate = fetch_macro_data('DFF', '2022-01-01', end)
    if not fed_funds_rate.empty:
        print("\n--- Fetched Federal Funds Rate Data ---")
        print(fed_funds_rate.tail())
