import pandas as pd
import logging
from joblib import load
from sklearn.metrics import recall_score
from pathlib import Path
import numpy as np

# libraries to import function from other folder
import sys
import os
sys.path.append(os.path.abspath('src/'))


from features.build_features import (print_shape, replace_missing_values,
    filter_cols_values, fill_na_with_col, fill_na_with_string, encoding_columns,
    medication_changes, medication_encoding, diagnose_encoding, to_categorical,
    drop_cols, encode_categorical, process_medical_specialty)

def main(input_filepath, output_filepath, input_test_filepath, report_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating ML model')

    model = load(f'{output_filepath}/NB_final_model.joblib')
    
    x_train = pd.read_csv(f"{input_filepath}/x_train_model_input.csv")
    y_train = pd.read_csv(f"{input_filepath}/y_train_model_input.csv")

    y_pred = model.predict(x_train)

    train_score = recall_score(y_train, y_pred)
    print(f"Train Score: {train_score}")

    with open(f'{report_filepath}/train_score.txt', 'w') as f:
        f.write(f"Train reacall Score: {train_score}")

    # test predictions

    x_test = pd.read_csv(f"{input_test_filepath}/x_test.csv")
    y_test = pd.read_csv(f"{input_test_filepath}/y_test.csv")

    test = pd.concat([x_test, y_test],axis = 1)

    test_eval = feature_process(test)

    x_test_model = test_eval.drop("readmitted", axis=1)
    y_test_model = test_eval["readmitted"]

    y_test_pred = model.predict(x_test_model)

    test_score = recall_score(y_test_model, y_test_pred)
    print(f"Test Score: {test_score}")

    with open(f'{report_filepath}/test_score.txt', 'w') as f:
        f.write(f"Test recall Score: {test_score}")    



def feature_process(data: pd.DataFrame) -> pd.DataFrame:

    """Process raw data into useful files for model."""
    cols_to_drop = ['encounter_id',
                      'patient_nbr',
                      'examide',
                      'citoglipton',
                      'glimepiride-pioglitazone',
                      'weight',
                      'payer_code',
                      'diag_3',
                      'gender'
                      ]
    medication = ['metformin', 'repaglinide', 'nateglinide', 
            'chlorpropamide', 'glimepiride', 'glipizide', 
            'glyburide', 'pioglitazone', 'rosiglitazone', 
            'acarbose', 'miglitol', 'insulin', 
            'glyburide-metformin', 'tolazamide', 
            'metformin-pioglitazone','metformin-rosiglitazone',
            'glipizide-metformin', 'troglitazone', 'tolbutamide',
            'acetohexamide']

    med_specialty = ['Unknown', 'InternalMedicine', 'Family/GeneralPractice',
                     'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                     'Nephrology', 'Orthopedics-Reconstructive',
                     'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                     'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                     'Urology', 'Surgery-Vascular', 'Radiologist']            

    cat_cols = ["admission_type_id", 
                "discharge_disposition_id",
                "admission_source_id"]                          

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(replace_missing_values, replace_values='?')
                    .pipe(print_shape, msg=' Shape after replace missing values')
                    .pipe(filter_cols_values, filter_col='discharge_disposition_id',
                          filter_values=[11, 13, 14, 19, 20])#test
                    .pipe(print_shape, msg=' Shape after filter discharge_disposition_id')
                    .pipe(filter_cols_values, filter_col='diag_1', filter_values=[np.nan])#test
                    .pipe(print_shape, msg=' Shape after filter diag_1')
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_3')#test
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_1')#test
                    .pipe(print_shape, msg=' Shape after fill diag_2')
                    .pipe(fill_na_with_string, fill_col='medical_specialty', fill_string='Unknown')#test
                    .pipe(print_shape, msg=' Shape after fill na medical specialty with Unknown')#test
                    .pipe(fill_na_with_string, fill_col='race', fill_string='Caucasian')#test
                    .pipe(encoding_columns)
                    .pipe(medication_changes,keys = medication)
                    .pipe(medication_encoding, keys = medication)
                    .pipe(diagnose_encoding)
                    .pipe(process_medical_specialty, keys = med_specialty)
                    .pipe(to_categorical, categorical_cols = cat_cols)
                    .pipe(drop_cols, drop_cols = cols_to_drop)
                    .pipe(print_shape, msg=' Shape after drop cols')
                    .pipe(encode_categorical)
                    .pipe(print_shape, msg=' Shape after encode categorical cols')
                    )
    return process_data                


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed', 
        f'{project_dir}/models',
        f'{project_dir}/data/interim', 
        f'{project_dir}/reports')