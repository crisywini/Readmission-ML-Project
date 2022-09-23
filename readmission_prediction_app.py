from typing import Dict
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

from src.features.build_features import (print_shape, replace_missing_values,
    filter_cols_values, fill_na_with_col, fill_na_with_string, encoding_columns,
    medication_changes, medication_encoding, diagnose_encoding, to_categorical,
    drop_cols, encode_categorical, process_medical_specialty)

# ---------------------- Lists ---------------------------
columns_list = [
            'encounter_id',
            'patient_nbr',
            'race',
            'gender',
            'age',
            'weight',
            'admission_type_id',
            'discharge_disposition_id',
            'admission_source_id',
            'time_in_hospital',
            'payer_code',
            'medical_specialty',
            'num_lab_procedures',
            'num_procedures',
            'num_medications',
            'number_outpatient',
            'number_emergency',
            'number_inpatient',
            'diag_1',
            'diag_2',
            'diag_3',
            'number_diagnoses',
            'max_glu_serum',
            'A1Cresult',
            'metformin',
            'repaglinide',
            'nateglinide',
            'chlorpropamide',
            'glimepiride',
            'acetohexamide',
            'glipizide',
            'glyburide',
            'tolbutamide',
            'pioglitazone',
            'rosiglitazone',
            'acarbose',
            'miglitol',
            'troglitazone',
            'tolazamide',
            'examide',
            'citoglipton',
            'insulin',
            'glyburide-metformin',
            'glipizide-metformin',
            'glimepiride-pioglitazone',
            'metformin-rosiglitazone',
            'metformin-pioglitazone',
            'change',
            'diabetesMed'
]

med_specialty = ['InternalMedicine', 'Family/GeneralPractice',
                 'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                 'Nephrology', 'Orthopedics-Reconstructive',
                 'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                 'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                 'Urology', 'Surgery-Vascular', 'Radiologist', 'Unknown', 'Other']

diag_1_list =['250.83', '276', '648', '8', '197', '414', '428', '398', '434',
       '250.7', '157', '518', '999', '410', '682', '402', '737', '572',
       'V57', '189', '786', '427', '996', '277', '584', '462', '473',
       '411', '174', '486', '998', '511', '432', '626', '295', '196',
       '250.6', '618', '182', '845', '423', '808', '250.4', '722', '403',
       '250.11', '784', '707', '440', '151', '715', '997', '198', '564',
       '812', '38', '590', '556', '578', '250.32', '433', 'V58', '569',
       '185', '536', '255', '250.13', '599', '558', '574', '491', '560',
       '244', '250.03', '577', '730', '188', '824', '250.8', '332', '562',
       '291', '296', '510', '401', '263', '438', '70', '250.02', '493',
       '642', '625', '571', '738', '593', '250.42', '807', '456', '446',
       '575', '250.41', '820', '515', '780', '250.22', '995', '235',
       '250.82', '721', '787', '162', '724', '282', '514', 'V55', '281',
       '250.33', '530', '466', '435', '250.12', 'V53', '789', '566',
       '822', '191', '557', '733', '455', '711', '482', '202', '280',
       '553', '225', '154', '441', '250.81', '349', '?', '962', '592',
       '507', '386', '156', '200', '728', '348', '459', '426', '388',
       '607', '337', '82', '531', '596', '288', '656', '573', '492',
       '220', '516', '210', '922', '286', '885', '958', '661', '969',
       '250.93', '227', '112', '404', '823', '532', '416', '346', '535',
       '453', '250', '595', '211', '303', '250.01', '852', '218', '782',
       '540', '457', '285', '431', '340', '550', '54', '351', '601',
       '723', '555', '153', '443', '380', '204', '424', '241', '358',
       '694', '331', '345', '681', '447', '290', '158', '579', '436',
       '335', '309', '654', '805', '799', '292', '183', '78', '851',
       '458', '586', '311', '892', '305', '293', '415', '591', '794',
       '803', '79', '655', '429', '278', '658', '598', '729', '585',
       '444', '604', '727', '214', '552', '284', '680', '708', '41',
       '644', '481', '821', '413', '437', '968', '756', '632', '359',
       '275', '512', '781', '420', '368', '522', '294', '825', '135',
       '304', '320', '250.31', '669', '868', '496', '250.43', '826',
       '567', '3', '203', '53', '251', '565', '161', '495', '49', '250.1',
       '297', '663', '576', '355', '850', '287', '250.2', '611', '840',
       '350', '726', '537', '620', '180', '366', '783', '11', '751',
       '716', '250.3', '199', '464', '580', '836', '664', '283', '813',
       '966', '289', '965', '184', '480', '608', '333', '972', '212',
       '117', '788', '924', '959', '621', '238', '785', '714', '942',
       '250.23', '710', '47', '933', '508', '478', '844', '7', '736',
       '233', '42', '250.5', '397', '395', '201', '421', '253', '250.92',
       '600', '494', '977', '39', '659', '312', '614', '647', '652',
       '646', '274', '861', '425', '527', '451', '485', '217', '250.53',
       '442', '970', '193', '160', '322', '581', '475', '623', '374',
       '582', '568', '465', '801', '237', '376', '150', '461', '913',
       '226', '617', '987', '641', '298', '790', '336', '362', '228',
       '513', '383', '746', '353', '911', '506', '873', '155', '860',
       '534', '802', '141', 'V45', '396', '310', '341', '242', '719',
       '239', '533', '616', '519', '301', 'V66', '5', '989', '230', '385',
       '300', '853', '871', '570', '848', '463', '9', '934', '250.21',
       '236', '361', '594', '501', '810', '643', '430', '528', '205',
       '791', '983', '992', '490', '172', '171', '622', '306', '863',
       '864', '474', '660', '759', '356', '634', '967', '551', '695',
       '187', '732', '747', '323', '308', '370', '252', '152', '846',
       '164', '365', '718', '48', '266', '720', '94', '344', '797', '170',
       '878', '904', 'V56', '882', '843', '709', '973', '454', '686',
       '939', '487', '229', '991', '483', '357', '692', '796', '693',
       '935', '936', '800', '920', 'V26', '261', '307', '262', '250.9',
       '831', '145', '223', 'V71', '839', '685', 'V54', '35', '34', '179',
       '964', '136', '324', '389', '815', '334', '143', '526', '588',
       '192', 'V67', '394', '917', '88', '219', '325', '792', '717',
       '994', '990', '793', '207', '637', '195', '373', '847', '827',
       '31', '891', '814', 'V60', '703', '865', '352', '627', '378',
       '342', '886', '369', '745', '705', '816', '541', '986', '610',
       '633', '640', '753', '173', '835', '379', '445', '272', '382',
       '945', '619', '881', '250.52', '866', '405', '916', '215', '893',
       '75', '671', '928', '906', '897', '725', '867', '115', '890',
       '734', '521', '674', '470', '834', '146', '696', '524', '980',
       '691', '384', '142', '879', '250.51', '246', '208', '448', '955',
       '653', '149', '245', '735', '883', '854', '952', '838', '194',
       'V43', '163', '216', '147', '354', '27', '477', '318', '880',
       '921', '377', '471', '683', '175', '602', '250.91', '982', '706',
       '375', '417', '131', '347', '870', '148', '862', '61', '817',
       '914', '360', '684', '314', 'V63', '36', '57', '240', '915', '971',
       '795', '988', '452', '963', '327', '731', '842', 'V25', '645',
       '665', '110', '944', '603', '923', '412', '363', '957', '976',
       '698', '299', '700', '273', '974', '97', '529', '66', '98', '605',
       '941', '52', '806', '84', '271', '837', '657', '895', '338', '523',
       '542', '114', '543', '372', 'V70', 'E909', '583', 'V07', '422',
       '615', '279', '500', '903', '919', '875', '381', '804', '704',
       '23', '58', '649', '832', '133', '975', '833', '391', '690', '10',
       'V51']

diag_2_list = ['250.01', '250', '250.43', '157', '411', '492', '427', '198',
       '403', '288', '998', '507', '174', '425', '456', '401', '715',
       '496', '428', '585', '250.02', '410', '999', '996', '135', '244',
       '41', '571', '276', '997', '599', '424', '491', '553', '707',
       '286', '440', '493', '242', '70', 'V45', '250.03', '357', '511',
       '196', '396', '197', '414', '250.52', '577', '535', '413', '285',
       '53', '780', '518', '150', '566', '250.6', '867', '486', 'V15',
       '8', '788', '340', '574', '581', '228', '530', '250.82', '786',
       '294', '567', '785', '512', '305', '729', '250.51', '280', '648',
       '560', '618', '444', '38', 'V10', '578', '277', '781', '250.42',
       '278', '426', '584', '462', '402', '153', '272', '733', '34',
       '881', '203', '250.41', '250.13', '293', '245', '250.12', '558',
       '787', '342', '573', '626', '303', '250.53', '458', '710', '415',
       'V42', '284', '569', '759', '682', '112', '292', '435', '290',
       '250.93', '642', '536', '398', '319', '711', 'E878', '446', '255',
       'V44', '250.7', '784', '300', '562', '162', '287', '447', '789',
       '790', '591', '200', '154', '304', '117', '847', '852', '250.83',
       '250.11', '816', '575', '416', '412', '441', '515', '372', '482',
       '382', 'V65', '572', '283', '78', '250.81', '576', '432', '595',
       '295', 'V12', '204', '466', '721', '434', '590', '271', '813',
       '368', '227', '783', '250.5', '258', '253', '309', '250.91', '519',
       '333', '459', '250.92', '250.4', '179', '420', '345', '433', '661',
       '537', '205', '722', '405', '437', '714', '211', 'E812', '263',
       '202', '397', '250.23', 'E932', '201', '301', '723', '614', '568',
       '861', 'V57', '724', '189', '297', '453', 'E888', '730', '354',
       '451', '738', 'E939', '805', 'V43', '155', '910', '218', '358',
       '220', 'E937', '583', '958', '794', '564', '436', '250.22', '620',
       '621', '331', '617', '596', '314', '378', '250.8', '625', '478',
       '731', '172', '404', '681', '470', '279', '281', '531', '443',
       '799', '337', '311', '719', 'E944', '423', 'E870', '465', 'E849',
       '782', '481', '480', 'V23', '199', '79', '438', '348', '42',
       'E950', '473', '627', '726', '54', '490', '317', '332', '508',
       '369', '600', '349', '485', '208', '922', '431', '296', 'E934',
       '753', 'E935', '386', '728', '607', 'E915', '344', '716', '289',
       '191', '873', '850', '611', '377', '352', '616', 'V17', '136',
       '455', '933', 'E885', '860', '513', '603', '484', '223', 'V72',
       '291', '151', 'V58', '550', '510', '891', '185', '592', '791',
       '138', '598', '336', '362', '217', '825', '298', '821', 'E880',
       '343', '429', 'E879', '579', '225', '250.9', 'V49', '696', '233',
       '658', '969', '275', '250.1', '601', '704', '808', 'E890', 'V18',
       '920', '380', '570', 'E817', '359', '812', '274', 'V14', '324',
       '758', 'V66', '911', 'E931', 'E924', '593', '792', '727', 'V46',
       '394', '532', 'V64', '557', '864', '718', 'E942', '807', '604',
       '924', '820', '580', '273', '241', '282', '824', 'V61', '646',
       '701', '736', '565', '383', '250.2', 'E947', '452', '872', '905',
       'E930', '921', '131', '448', '389', '421', '214', '705', '494',
       '752', '623', '9', '299', '959', '365', '967', 'E858', '40', '691',
       '909', '5', '814', '746', '250.31', '556', '680', '745', '351',
       '306', '110', '695', '552', '346', '918', '882', '947', '520',
       '188', '31', '356', '737', 'V08', '322', '182', '517', '974',
       'E929', 'V53', '912', '252', '608', '516', 'E933', '94', '702',
       '923', '594', '647', '111', '934', '430', '487', '709', '796',
       '156', '977', '915', '756', '840', '341', '259', '693', '725',
       'V62', '528', '683', '953', '457', '501', 'E900', 'V09', '522',
       '919', '461', '506', '193', '483', 'E936', '717', '802', '335',
       'V54', '320', '945', '906', '239', '454', '826', '823', 'E941',
       '226', '795', '684', '844', '250.33', '308', '615', '588', '712',
       '663', '706', '833', '741', '713', '533', 'E884', '586', '555',
       '755', 'E928', '742', '869', '962', 'V11', '543', '373', '870',
       '913', '152', '810', '965', '907', '908', '995', '845', '474',
       '442', '751', '323', '472', '464', '686', '250.32', '540', '251',
       '811', '652', '659', '851', '422', '815', '307', '325', '463',
       '992', '692', '521', '917', 'E965', '524', '916', 'E813', '173',
       '238', '137', '514', '312', '837', '355', '980', '622', '475',
       '500', '754', '261', '801', '868', '968', '381', '11', '250.21',
       '694', '610', '734', 'E814', '310', '130', '246', '892', '846',
       '634', '75', 'E927', 'E905', '183', '379', 'E917', '163', 'E868',
       '495', '747', '989', 'E854', '240', '832', '605', '602', '644',
       'V16', '35', 'V70', '376', '266', 'E918', '619', '477', '656',
       '46', '883', '171', 'V13', '698', '842', 'E850', '800', '269',
       '664', 'E887', '952', '164', 'E881', '527', '685', '366', '836',
       '27', 'V63', '865', '793', '232', '990', '52', '831', '327', '542',
       '806', '972', '862', 'E829', 'E919', '944', 'E916', '963', '316',
       '645', '347', 'V85', '374', 'V02', '748', '256', '186', '866',
       '975', '96', '395', '262', 'E819', '654', '994', '318', 'E826',
       '879', '674', '641', '822', '145', '797', '353', 'E938', 'E816',
       '948', '987', '99', '192', '250.3', 'E906', '534', '115', 'E818',
       'E980', '360', '338', '529', '871', '750', '212', '302', '955',
       '141', '88', 'V25', '215', '350', 'V50', 'V03', 'E853', 'E968',
       'E882', '140', '703', '991', '893', 'E821', '235', 'V69', '670',
       '195', 'V55', '388', '268', '894', '114', '260', '853', '7', '880',
       'V86', '180', 'E945', '523', '863', '649', '270', '665', '460',
       '942', '364', '66', 'E883', '123', '884', 'V60', '843', '927', '?'] 

# ---------------------- Prediction Model ------------------------------

model = load("models/NB_final_model.joblib")

#Caching the model for faster loading
@st.cache
# list of inputs are too long so i use *args instead
def predict(*args):
    list_variables = list(args)
    data_input = pd.DataFrame([list_variables],columns= columns_list)
    data_predict = feature_process(data_input)

    prediction = model.predict(data_predict)
    return prediction

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
                    .pipe(replace_missing_values, replace_values='?')
                    .pipe(filter_cols_values, filter_col='discharge_disposition_id',
                          filter_values=[11, 13, 14, 19, 20])#test
                    .pipe(filter_cols_values, filter_col='diag_1', filter_values=[np.nan])#test
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_3')#test
                    .pipe(fill_na_with_col, fill_col='diag_2', fill_col_from='diag_1')#test
                    .pipe(fill_na_with_string, fill_col='medical_specialty', fill_string='Unknown')#test
                    .pipe(fill_na_with_string, fill_col='race', fill_string='Caucasian')#test
                    .pipe(encoding_columns)
                    .pipe(medication_changes,keys = medication)
                    .pipe(medication_encoding, keys = medication)
                    .pipe(diagnose_encoding)
                    .pipe(process_medical_specialty, keys = med_specialty)
                    .pipe(to_categorical, categorical_cols = cat_cols)
                    .pipe(drop_cols, drop_cols = cols_to_drop)
                    .pipe(encode_categorical)
                    .pipe(add_missing_cols)
                    )
    return process_data

def add_missing_cols(data: pd.DataFrame) -> pd.DataFrame:
    trains_cols = [
                'admission_type_id',
                'discharge_disposition_id',
                'admission_source_id',
                'time_in_hospital',
                'num_lab_procedures',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_emergency',
                'number_inpatient',
                'number_diagnoses',
                'max_glu_serum',
                'A1Cresult',
                'metformin',
                'repaglinide',
                'nateglinide',
                'chlorpropamide',
                'glimepiride',
                'acetohexamide',
                'glipizide',
                'glyburide',
                'tolbutamide',
                'pioglitazone',
                'rosiglitazone',
                'acarbose',
                'miglitol',
                'troglitazone',
                'tolazamide',
                'insulin',
                'glyburide-metformin',
                'glipizide-metformin',
                'metformin-rosiglitazone',
                'metformin-pioglitazone',
                'change',
                'diabetesMed',
                'numchange',
                'race_Asian',
                'race_Caucasian',
                'race_Hispanic',
                'race_Other',
                'age_[10-20)',
                'age_[20-30)',
                'age_[30-40)',
                'age_[40-50)',
                'age_[50-60)',
                'age_[60-70)',
                'age_[70-80)',
                'age_[80-90)',
                'age_[90-100)',
                'medical_specialty_Emergency/Trauma',
                'medical_specialty_Family/GeneralPractice',
                'medical_specialty_Gastroenterology',
                'medical_specialty_InternalMedicine',
                'medical_specialty_Nephrology',
                'medical_specialty_ObstetricsandGynecology',
                'medical_specialty_Orthopedics',
                'medical_specialty_Orthopedics-Reconstructive',
                'medical_specialty_Other',
                'medical_specialty_Psychiatry',
                'medical_specialty_Pulmonology',
                'medical_specialty_Radiologist',
                'medical_specialty_Surgery-Cardiovascular/Thoracic',
                'medical_specialty_Surgery-General',
                'medical_specialty_Surgery-Neuro',
                'medical_specialty_Surgery-Vascular',
                'medical_specialty_Unknown',
                'medical_specialty_Urology',
                'diag_1_Diabetes',
                'diag_1_Digestive',
                'diag_1_Genitourinary',
                'diag_1_Injury',
                'diag_1_Muscoloskeletal',
                'diag_1_Neoplasms',
                'diag_1_Others',
                'diag_1_Respiratory',
                'diag_2_Diabetes',
                'diag_2_Digestive',
                'diag_2_Genitourinary',
                'diag_2_Injury',
                'diag_2_Muscoloskeletal',
                'diag_2_Neoplasms',
                'diag_2_Others',
                'diag_2_Respiratory'
            ]
    missing_cols = set( trains_cols ) - set( data.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    data = data[trains_cols]
    return data                    

# --------------- streamlit app ------------------------------------
st.title('Diabetes readmission prediction')
st.image("""https://storage.googleapis.com/kaggle-datasets-images/3724/5903/a8a637953c923bf989852df53b54d769/dataset-card.jpg""")
st.header('Insert the patient information')
race = st.radio('Race:', ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'], horizontal = True)
age = st.radio('Age:', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
       '[60-70)', '[70-80)', '[80-90)', '[90-100)'], horizontal = True)
admission_type_id = st.number_input('Admission Type:',min_value=1, max_value=8, value=1)
discharge_disposition_id = st.number_input('Discharge disposition:',min_value=1, max_value=28, value=1)
admission_source_id = st.number_input('Admission Source:',min_value=1, max_value=25, value=1)
time_in_hospital = st.number_input('time in hospital:',min_value=1, max_value=100, value=1)
medical_specialty = st.selectbox('Medical Specialty:', med_specialty)
num_lab_procedures = st.number_input('# lab Procedures:',min_value=0, max_value=100, value=0)
num_procedures = st.number_input('# Procedures:',min_value=0, max_value=100, value=0)
num_medications = st.number_input('# Medications:',min_value=0, max_value=100, value=0)
number_outpatient = st.number_input('# outpatient visits:',min_value=0, max_value=100, value=0)
number_emergency = st.number_input('# of emergency visits:',min_value=0, max_value=100, value=0)
number_inpatient = st.number_input('# inpatient visits:',min_value=0, max_value=100, value=0)
diag_1 = st.selectbox('Diagnosis 1', diag_1_list)
diag_2 = st.selectbox('Diagnosis 2', diag_2_list)
diag_3 = '250' #random
number_diagnoses = st.number_input('# of diagnosis:',min_value=1, max_value=100, value=1)
max_glu_serum = st.radio('Glucose serum test result:', ['None', '>300', 'Norm', '>200'], horizontal = True)
A1Cresult = st.radio('A1c test result:', ['None', '>8', '>7', 'Norm' ], horizontal = True)
metformin = st.radio('metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
repaglinide = st.radio('repaglinide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
nateglinide = st.radio('nateglinide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
chlorpropamide = st.radio('chlorpropamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glimepiride = st.radio('glimepiride', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
acetohexamide = st.radio('acetohexamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glipizide = st.radio('glipizide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glyburide = st.radio('glyburide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
tolbutamide = st.radio('tolbutamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
pioglitazone = st.radio('pioglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
rosiglitazone = st.radio('rosiglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
acarbose = st.radio('acarbose', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
miglitol = st.radio('miglitol', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
troglitazone = st.radio('troglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
tolazamide = st.radio('tolazamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
insulin = st.radio('insulin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glyburide_metformin = st.radio('glyburide-metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glipizide_metformin = st.radio('glipizide-metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
metformin_rosiglitazone = st.radio('metformin-rosiglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
metformin_pioglitazone = st.radio('metformin-pioglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)

change = st.radio('Chagne of medications', ['No', 'Ch'] , horizontal = True)
diabetesMed = st.radio('Diabetes medications', ['no', 'yes'], horizontal = True)

#columns to drop
encounter_id = 54500028 #random number
patient_nbr = 3851154 #random number
gender = 'Female' # random
payer_code = 'MC' # random
examide = 'No' # random
citoglipton = 'No' # random
glimepiride_pioglitazone = 'No' # random
weight = 'No' # random


#data_input = dict(zip(columns_list,columns_variables))

if st.button('Predict Readmission'):
    readmission = predict(
            encounter_id,
            patient_nbr,
            race,
            gender,
            age,
            weight,
            admission_type_id,
            discharge_disposition_id,
            admission_source_id,
            time_in_hospital,
            payer_code,
            medical_specialty,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            diag_1,
            diag_2,
            diag_3,
            number_diagnoses,
            max_glu_serum,
            A1Cresult,
            metformin,
            repaglinide,
            nateglinide,
            chlorpropamide,
            glimepiride,
            acetohexamide,
            glipizide,
            glyburide,
            tolbutamide,
            pioglitazone,
            rosiglitazone,
            acarbose,
            miglitol,
            troglitazone,
            tolazamide,
            examide,
            citoglipton,
            insulin,
            glyburide_metformin,
            glipizide_metformin,
            glimepiride_pioglitazone,
            metformin_rosiglitazone,
            metformin_pioglitazone,
            change,
            diabetesMed
    )
    st.success(f'The prediction tells that the patient will have a readmission in less than 30 days: {bool(readmission)}')