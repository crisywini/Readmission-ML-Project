import pandas as pd
import numpy as np


# Functions for reading, processing, and writing data from the UCI
# Diabetes 130-US hospitals for years 1999-2008 Data Set dataset.

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""
    cols_to_remove = ['encounter_id',
                      'patient_nbr',
                      'examide',
                      'citoglipton',
                      'glimepiride-pioglitazone',
                      'weight',
                      'payer_code',
                      'diag_3',
                      'gender'
                      ]

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_duplicates, drop_cols=['patient_nbr'])
                    .pipe(print_shape, msg=' Shape after drop duplicates')
                    .pipe(replace_missing_values, replace_values='?')
                    .pipe(print_shape, msg=' Shape after replace missing values')
                    .pipe(filter_cols_values, filter_col='discharge_disposition_id',
                          filter_values=[11, 13, 14, 19, 20])
                    .pipe(print_shape, msg=' Shape after filter discharge_disposition_id')
                    .pipe(filter_cols_values, filter_col='diag_1', filter_values=[np.nan])
                    .pipe(print_shape, msg=' Shape after filter diag_1')
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_3')
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_1')
                    .pipe(print_shape, msg=' Shape after fill diag_2')
                    .pipe(fill_na_with_string, fill_col='medical_specialty', fill_string='Unknown')
                    .pipe(print_shape, msg=' Shape after fill na medical specialty with Unknown')
                    .pipe(drop_cols, drop_cols=cols_to_remove)
                    .pipe(print_shape, msg=' Shape after drop columns')
                    )

    return process_data


# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data


# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list | None) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(subset=drop_cols, keep=False)


# funtion to remove columns from data
def drop_cols(data: pd.DataFrame,
              drop_cols: list = None) -> pd.DataFrame:
    """Drop columns from data."""
    return data.drop(drop_cols, axis=1)


# function to filter data from a column if is in a list of values
def filter_cols_values(data: pd.DataFrame,
                       filter_col: str,
                       filter_values: list) -> pd.DataFrame:
    """Filter columns from data."""
    data = data[~data[filter_col].isin(filter_values)]
    return data


# function to replace ? values with np.nan
def replace_missing_values(data: pd.DataFrame,
                           replace_values: str) -> pd.DataFrame:
    """Replace missing values in data with np.nan"""
    return data.replace(replace_values, np.nan)


# function to fillna with a string in a column
def fill_na_with_string(data: pd.DataFrame,
                        fill_col: str,
                        fill_string: str) -> pd.DataFrame:
    """Fill missing values in data with a string."""
    data[fill_col] = data[fill_col].fillna(fill_string, axis=0)
    return data

# function to fill na values in one column using the values from another column
def fill_na_with_col(data: pd.DataFrame,
                     fill_col: str,
                     fill_col_from: str) -> pd.DataFrame:
        """Fill missing values in data with values from another column."""
        data[fill_col] = data[fill_col].fillna(data[fill_col_from], axis=0)
        return data


# def write_data(self, processed_data_path):
#     """Write processed data to directory."""
#     do writing things
