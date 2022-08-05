# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from preprocessing import process_data



def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info('reading data')
    data_raw = pd.read_csv(f"{input_filepath}/diabetic_data.csv")
    
    logger.info('processing data')
    processed_data = process_data(data_raw)

    print(f'ready data = {processed_data.shape}')

    logger.info('saving processed data')
    processed_data.reset_index(inplace=True, drop=True)
    processed_data.to_csv(f'{output_filepath}/diabetic_data_clean.csv',index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(f'{project_dir}/data/raw', f'{project_dir}/data/interim')

   
