import pytest
import pandas as pd
import numpy as np

#import created functions
from src.features.build_features import encoding_columns, diagnose_encoding, \
    process_medical_specialty, medication_encoding


#load data
@pytest.fixture
def leer_datos():
    data_test = pd.read_csv('data/interim/x_train.csv')
    return data_test


def test_data(leer_datos):
    columnas_name = ['encounter_id','patient_nbr',
                'race','gender','age','weight',
                'admission_type_id','discharge_disposition_id',
                'admission_source_id','time_in_hospital',
                'payer_code','medical_specialty',
                'num_lab_procedures','num_procedures',
                'num_medications','number_outpatient',
                'number_emergency','number_inpatient',
                'diag_1','diag_2','diag_3','number_diagnoses',
                'max_glu_serum','A1Cresult','metformin',
                'repaglinide','nateglinide','chlorpropamide',
                'glimepiride','acetohexamide','glipizide',
                'glyburide','tolbutamide','pioglitazone',
                'rosiglitazone','acarbose','miglitol',
                'troglitazone','tolazamide','examide',
                'citoglipton','insulin','glyburide-metformin',
                'glipizide-metformin','glimepiride-pioglitazone',
                'metformin-rosiglitazone','metformin-pioglitazone',
                'change','diabetesMed']
    columnas = leer_datos.columns
    assert len(columnas) == 49
    assert set(columnas_name) == set(columnas)


def test_encoding_columns(leer_datos):
    data = encoding_columns(leer_datos)

    assert list(data["max_glu_serum"].unique()) == [-99, 1, 0]
    assert list(data["A1Cresult"].unique()) == [-99, 1, 0]
    assert list(data['change'].unique()) == [0, 1]
    assert list(data['gender'].unique()) == [0, 1, 'Unknown/Invalid']
    assert list(data['diabetesMed'].unique()) == [0, 1]

def test_diagnose_encoding(leer_datos):

    datos = leer_datos.replace('?', np.nan)# replace ? with nan
    datos.dropna(inplace=True)# dropna values
    data = diagnose_encoding(datos)

    assert data["diag_1"].nunique() == 9
    assert data["diag_2"].nunique() == 9

def test_process_medical_specialty(leer_datos):
    med_specialty = ['Unknown', 'InternalMedicine', 'Family/GeneralPractice',
                     'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                     'Nephrology', 'Orthopedics-Reconstructive',
                     'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                     'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                     'Urology', 'Surgery-Vascular', 'Radiologist']
    data = process_medical_specialty(leer_datos, med_specialty)
    medical_specialty_list = list(data['medical_specialty'].unique())
    assert 'Other' in medical_specialty_list

def test_medication_encoding(leer_datos):
    medication = ['metformin', 'repaglinide', 'nateglinide',
            'chlorpropamide', 'glimepiride', 'glipizide',
            'glyburide', 'pioglitazone', 'rosiglitazone',
            'acarbose', 'miglitol', 'insulin',
            'glyburide-metformin', 'tolazamide',
            'metformin-pioglitazone','metformin-rosiglitazone',
            'glipizide-metformin', 'troglitazone', 'tolbutamide',
            'acetohexamide']
    datos = leer_datos.replace('?', np.nan)# replace ? with nan
    datos.dropna(inplace=True)# dropna values
    data = medication_encoding(datos, medication)

    assert list(data[medication[0]].unique()) == [0, 1]
    assert list(data[medication[1]].unique()) == [1, 0]
    assert list(data[medication[2]].unique()) == [0, 1]
    assert list(data[medication[3]].unique()) == [0]
    assert list(data[medication[4]].unique()) == [0,1]
    assert list(data[medication[5]].unique()) == [0, 1]





