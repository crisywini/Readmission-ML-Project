import pandas as pd
import numpy as np


# Functions for reading, processing, and writing data from the UCI
# Diabetes 130-US hospitals for years 1999-2008 Data Set dataset.

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after remove exact duplicates')
                    .pipe(transform_output) 
                    .pipe(sort_data, col = 'readmitted')
                    .pipe(drop_duplicates, drop_cols=['patient_nbr'])
                    .pipe(print_shape, msg=' Shape after remove patient duplicates')                             
                    )

    return process_data
# remove duplicates by patient_nbr in this step to avoid 
# same person in train and test

# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

def sort_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    "Sort data by and specific column"
    data = data.sort_values(by=col,ascending= False)
    return data

# remove duplicates from data based on a column
def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')    
    return data

def transform_output(data: pd.DataFrame) -> pd.DataFrame:
    """ Replace target column to 1 and 0 values"""

    data["readmitted"].replace({'<30':1,
                                '>30':0,
                                'NO':0},
                                inplace=True)
    return data                                