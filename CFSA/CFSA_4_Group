mport sys
import os
import math
import argparse
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from numpy import mean, std
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from Bayesian_fix_four_group import data_biased_dis
from sklearn import  linear_model
from pandas.core.frame import DataFrame

def get_data(dataset_used, protected, preprocessed = False):
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
        #'hours-per-week', capital-gain','capital-loss','education-num'
        columnname = ['age','education-num']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)

    # elif dataset_used == "crime":


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

    elif dataset_used == "dutch":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/dutch.csv')
        dataset_orig.rename(columns={'occupation': 'Probability'}, inplace=True)
        dataset_orig = dataset_orig.dropna()
        dataset_orig = dataset_orig.drop(
            ['prev_residence_place','citizenship','country_birth'], axis=1)
        mean = dataset_orig.loc[:, "age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'male', 1, 0)
        columnname = ['household_size','economic_status', 'marital_status','cur_eco_activity']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

    elif dataset_used == "ricci":
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = pd.read_csv('data/ricci_race.csv')
        dataset_orig.rename(columns={'Promoted': 'Probability'}, inplace=True)
        dataset_orig = dataset_orig.dropna()
        dataset_orig = dataset_orig.drop(['Position'], axis=1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'White', 0, 1)

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


    elif dataset_used == "titanic":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/titanic.csv')
        dataset_orig = dataset_orig.dropna()
        dataset_orig.rename(index=str, columns={'Survived': 'Probability'}, inplace=True)
        dataset_orig = dataset_orig.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Fare','SibSp'], axis=1)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'male', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 1, 0)
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
        columnname = ['age','Embarked']
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
        # columnname = ['fam_inc','tier']
        # for i in columnname:
        #     status_dict = dataset_orig[i].unique().tolist()
        #     dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        print(dataset_orig)

    elif dataset_used == "student":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/Student.csv')
        dataset_orig = dataset_orig.drop(['address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','reason','guardian','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','G1','G2'], axis=1)
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 12, 1, 0)
        columnname = ['school','age','traveltime','studytime','absences']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

    elif dataset_used == "german":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/GermanData.csv')
        dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'], axis=1)
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
        columnname = ['credit_history','savings','employment','age']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)


    elif dataset_used == "bank":
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        # dataset_orig = BankDataset().convert_to_dataframe()[0]
        dataset_orig = pd.read_csv('data/bank.csv')
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 30, 1, 0)
        dataset_orig['House'] = np.where(dataset_orig['House'] == 'yes', 1, 0)
        dataset_orig['month'] = np.where(dataset_orig['month'] == 'May', 5, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)
        columnname = ['job','marital','education','default','loan','contact','poutcome']
        for i in columnname:
            status_dict = dataset_orig[i].unique().tolist()
            dataset_orig[i] = dataset_orig[i].apply(lambda x: status_dict.index(x))

        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)

    return dataset_orig, privileged_groups,unprivileged_groups

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

def data_dis(dataset_orig_test,protected_attribute,dataset_used):

    if dataset_used == 'bank':
        zero_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
    else:
        zero_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])
    print(zero_zero,zero_one, one_zero, one_one)
    a=zero_one+one_one
    b=-1*(zero_zero*zero_one+2*zero_zero*one_one+one_zero*one_one)
    c=(zero_zero+one_zero)*(zero_zero*one_one-zero_one*one_zero)
    x=abs((-b-math.sqrt(b*b-4*a*c))/(2*a))
    y=(zero_one+one_one)/(zero_zero+one_zero)*x
    zero_zero_new = int(zero_zero-x)
    one_one_new = int(one_one-y)

    print(zero_zero_new, one_one_new, x, y)
    if dataset_used == 'bank':
        zero_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)]
        zero_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)].sample(n=zero_zero_new)
        one_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)].sample(n=one_one_new)
        new_set = zero_zero_set.append([zero_one_set, one_zero_set, one_one_set], ignore_index=True)
    else:
        zero_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)]
        zero_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)].sample(
            n=zero_zero_new)
        one_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)].sample(
            n=one_one_new)
        new_set = zero_zero_set.append([zero_one_set, one_zero_set, one_one_set], ignore_index=True)

    zero_zero = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 0)])
    zero_one = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 1)])
    one_zero = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 0)])
    one_one = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 1)])

    print("distribution:", zero_zero, zero_one, one_zero, one_one)
    print(zero_zero/(zero_zero+one_zero), zero_one/(zero_one+one_one))

    return new_set


def measure_final_score(dataset_orig_test, dataset_orig_predict,privileged_groups,unprivileged_groups):

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    spd = abs(classified_metric_pred.statistical_parity_difference())
    aaod = classified_metric_pred.average_abs_odds_difference()
    eod = abs(classified_metric_pred.equal_opportunity_difference())

    return accuracy, recall_macro,  precision_macro,  f1score_macro, mcc, spd, aaod, eod


# dataset_used = 'bank'
# attr = 'age'
# clf_name = 'lr'

# dataset_used = 'bank'
# attr = 'age'
# clf_name = 'svm'

#Adult sex
dataset_used = 'adult'
attr = 'sex'
clf_name = 'lr'

#Adult race
# dataset_used = 'adult'
# attr = 'race'
# clf_name = 'lr'

#Compas
# dataset_used = 'compas'
# attr = 'race'
# clf_name = 'lr'

# dataset_used = 'compas'
# attr = 'sex'
# clf_name = 'lr'

#German
# dataset_used = 'german'
# attr = 'sex'
# clf_name = 'lr'

# dataset_used = 'german'
# attr = 'sex'
# clf_name = 'svm'
#Student
# dataset_used = 'student'
# attr = 'sex'
# clf_name = 'lr'

# dataset_used = 'student'
# attr = 'sex'
# clf_name = 'lr'

#Heart
# dataset_used = 'heart'
# attr = 'sex'
# clf_name = 'lr'

# dataset_used = 'heart'
# attr = 'sex'
# clf_name = 'svm'

#ricci
# dataset_used = 'ricci'
# attr = 'race'
# clf_name = 'svm'

# dataset_used = 'ricci'
# attr = 'race'
# clf_name = 'lr'

#Law
# dataset_used = 'law'
# attr = 'race'
# clf_name = 'lr'

# dataset_used = 'law'
# attr = 'sex'
# clf_name = 'lr'

#titanic
# dataset_used = 'titanic'
# attr = 'sex'
# clf_name = 'lr'

#Default
# dataset_used = 'default'
# attr = 'age'
# clf_name = 'lr'

# dataset_used = 'default'
# attr = 'sex'
# clf_name = 'lr'
#Dutch
# dataset_used = 'dutch'
# attr = 'sex'
# clf_name = 'lr'

# dataset_used = 'dutch'
# attr = 'sex'
# clf_name = 'svm'

# dataset_used = 'dutch'
# attr = 'sex'
# clf_name = 'rf'

val_name = "1_0_Bayesian_four_group_maat_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aaod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 50

for r in range(repeat_time):
    print (r)
    if (r == 0):
        np.random.seed(r)
        #split training data and test data
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

        dataset_orig_train_new = data_dis(pd.DataFrame(dataset_orig_train), attr, dataset_used)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new)
        dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                                protected_attribute_names=[attr])
        dataset_orig_train_new = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new,label_names=['Probability'],protected_attribute_names=[attr])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,label_names=['Probability'],protected_attribute_names=[attr])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,label_names=['Probability'],protected_attribute_names=[attr])
        clf = get_classifier(clf_name)
        if clf_name == 'lr':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf1 = clf.fit(dataset_orig_train.features, np.ravel(dataset_orig_train.labels))
        clf = get_classifier(clf_name)
        if clf_name == 'lr':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf2 = clf.fit(dataset_orig_train_new.features, np.ravel(dataset_orig_train_new.labels))
        pred_de1 = clf1.predict_proba(dataset_orig_test_1.features)
        print("pred_de1",pred_de1)
        pred_de2 = clf2.predict_proba(dataset_orig_test_2.features)

 

        test_df_copy = copy.deepcopy(dataset_orig_test_1)


        res = []
        p1 = []
        p2 = []
        for i in range(len(pred_de1)):
            prob_t = 1*pred_de1[i][1]+0*pred_de2[i][1]
            p1.append(pred_de1[i][1])
            p2.append(pred_de2[i][1])
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)
        lin_reg = linear_model.LinearRegression()
        prob_lin = {"x1":p1,"x2":p2}
        prob_lin = DataFrame(prob_lin)
        lin_reg.fit(prob_lin, res)
        w = lin_reg.coef_
        b = lin_reg.intercept_
        print(w,b)
        test_df_copy.labels = np.array(res)

        round_result= measure_final_score(dataset_orig_test_1,test_df_copy,privileged_groups,unprivileged_groups)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])
    else:
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        dataset_orig_train_new = data_biased_dis(pd.DataFrame(dataset_orig_train), attr, dataset_used)
        print(type(dataset_orig_train_new))
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new)
        dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                                label_names=['Probability'],
                                                protected_attribute_names=[attr])
        dataset_orig_train_new = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new,
                                                    label_names=['Probability'], protected_attribute_names=[attr])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                                 label_names=['Probability'], protected_attribute_names=[attr])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                                 label_names=['Probability'], protected_attribute_names=[attr])
        clf = get_classifier(clf_name)
        if clf_name == 'lr':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf1 = clf.fit(dataset_orig_train.features, np.ravel(dataset_orig_train.labels))
        clf = get_classifier(clf_name)
        if clf_name == 'lr':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf2 = clf.fit(dataset_orig_train_new.features, np.ravel(dataset_orig_train_new.labels))
        pred_de1 = clf1.predict_proba(dataset_orig_test_1.features)
        # print("pred_de1", pred_de1)
        pred_de2 = clf2.predict_proba(dataset_orig_test_2.features)
        test_df_copy = copy.deepcopy(dataset_orig_test_1)

        res = []
        for i in range(len(pred_de1)):
            prob_t = 1*pred_de1[i][1]+0*pred_de2[i][1]
            # prob_t = pred_de1[i][1] * w[0] + pred_de2[i][1] * w[1] + b
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)

        test_df_copy.labels = np.array(res)

        round_result = measure_final_score(dataset_orig_test_1, test_df_copy, privileged_groups, unprivileged_groups)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
fout.close()
