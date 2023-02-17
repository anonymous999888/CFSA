from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
import numpy as np
import pandas as pd

# protected in {sex,race,age}
def get_data(dataset_used, protected,preprocessed = False):
    if dataset_used == "adult":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = pd.read_csv('data/adult.data.csv')
        # dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
        dataset_orig = dataset_orig.dropna()
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 60) & (dataset_orig['age'] < 70), 60,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 50) & (dataset_orig['age'] < 60), 50,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 40) & (dataset_orig['age'] < 50), 40,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 30) & (dataset_orig['age'] < 40), 30,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 20) & (dataset_orig['age'] < 30), 20,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 10) & (dataset_orig['age'] < 10), 10,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])
        dataset_orig = dataset_orig.drop(
            ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'],
            axis=1)
        # columnname = ['workclass','education','marital-status','occupation','relationship','native-country']
        # 'hours-per-week', capital-gain','capital-loss','education-num'
        columnname = ['age', 'education-num']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
    elif dataset_used == "law":
        # privileged_groups = [{'race': 1}]
        # unprivileged_groups = [{'race': 0}]
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/law_school_clean.csv')
        dataset_orig.rename(columns={'male': 'sex'}, inplace=True)
        dataset_orig = dataset_orig.drop(['decile1b','decile3','zfygpa','zgpa','fulltime','fam_inc'], axis=1)
        dataset_orig.rename(index=str, columns={'pass_bar': 'Probability'}, inplace=True)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'White', 0, 1)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 1, 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1.0, 1, 0)
    elif dataset_used == "default":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        dataset_orig = pd.read_csv('data/default_of_credit_card_clients_first_row_removed.csv')
        dataset_orig = dataset_orig.dropna()
        #'MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
        dataset_orig = dataset_orig.drop(['ID','LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis=1)
        dataset_orig.rename(index=str, columns={'default payment next month': 'Probability'}, inplace=True)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0, 1)
        mean = dataset_orig.loc[:, "age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 1, 0)
    elif dataset_used == "german":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/GermanData.csv')
        dataset_orig = dataset_orig.drop(
            ['1', '2', '4', '5', '8', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20'], axis=1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 60) & (dataset_orig['age'] < 70), 60,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 50) & (dataset_orig['age'] < 60), 50,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 40) & (dataset_orig['age'] < 50), 40,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 30) & (dataset_orig['age'] < 40), 30,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 20) & (dataset_orig['age'] < 30), 20,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 10) & (dataset_orig['age'] < 10), 10,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])
        columnname = ['credit_history', 'savings', 'employment', 'age']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
    elif dataset_used == "dutch":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/dutch.csv')
        dataset_orig.rename(columns={'occupation': 'Probability'}, inplace=True)
        dataset_orig = dataset_orig.dropna()
        dataset_orig = dataset_orig.drop(
            ['prev_residence_place', 'citizenship', 'country_birth'], axis=1)
        mean = dataset_orig.loc[:, "age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'male', 1, 0)
        columnname = ['household_size', 'economic_status', 'marital_status', 'cur_eco_activity']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))
    elif dataset_used == "heart":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/processed.cleveland.data.csv')

        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 0, 1, 0)
        mean = dataset_orig.loc[:, "age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        columnname = ['cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))
    elif dataset_used == "compas":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = pd.read_csv('data/compas-scores-two-years.csv')
        dataset_orig = dataset_orig.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age', 'juv_fel_count', 'decile_score',
             'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text', 'screening_date',
             'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
             'start', 'end', 'event'], axis=1)

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()
        dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
        dataset_orig['priors_count'] = np.where(
            (dataset_orig['priors_count'] >= 1) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
        dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45', 45, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
        dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)
        columnname = ['age_cat', 'priors_count']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))


    elif dataset_used == "bank":

        privileged_groups = [{'age': 1}]

        unprivileged_groups = [{'age': 0}]

        # dataset_orig = BankDataset().convert_to_dataframe()[0]

        dataset_orig = pd.read_csv('data/bank.csv')

        dataset_orig['age'] = np.where(dataset_orig['age'] >= 30, 1, 0)

        dataset_orig['House'] = np.where(dataset_orig['House'] == 'yes', 1, 0)

        dataset_orig['month'] = np.where(dataset_orig['month'] == 'May', 5, 0)

        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)

        columnname = ['job', 'marital', 'education', 'default', 'loan', 'contact', 'poutcome']

        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()

            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
    elif dataset_used == "mep":
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
    return dataset_orig, privileged_groups,unprivileged_groups


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf
