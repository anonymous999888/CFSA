import pandas as pd
import random
import time
import csv
import numpy as np
import math
import copy
import os

#Adult dataset visualization

adult_df = pd.read_csv('data/adult.data.csv')
adult_df = adult_df.dropna()


adult_df['sex'] = np.where(adult_df['sex'] == ' Male', 1, 0)
adult_df['race'] = np.where(adult_df['race'] != ' White', 0, 1)
adult_df['Probability'] = np.where(adult_df['Probability'] == ' <=50K', 0, 1)



#Based on class
adult_df_one , adult_df_zero = [x for _, x in adult_df.groupby(adult_df['Probability'] == 0)]

#Based on sex
adult_df_one_male, adult_df_one_female = [x for _, x in adult_df_one.groupby(adult_df_one['sex'] == 0)]
adult_df_zero_male, adult_df_zero_female = [x for _, x in adult_df_zero.groupby(adult_df_zero['sex'] == 0)]

#Based on race
adult_df_one_white, adult_df_one_nonwhite = [x for _, x in adult_df_one.groupby(adult_df_one['race'] == 0)]
adult_df_zero_white, adult_df_zero_nonwhite = [x for _, x in adult_df_zero.groupby(adult_df_zero['race'] == 0)]

print(adult_df_one_male.shape,adult_df_one_female.shape,adult_df_zero_male.shape,adult_df_zero_female.shape)
print(adult_df_one_white.shape,adult_df_one_nonwhite.shape,adult_df_zero_white.shape,adult_df_zero_nonwhite.shape)
print(adult_df_one.shape , adult_df_zero.shape)
print(adult_df.shape)
#compas dataset visualization

compas_df = pd.read_csv('data/compas-scores-two-years.csv')


compas_df['sex'] = np.where(compas_df['sex'] == 'Female', 1, 0)
compas_df['race'] = np.where(compas_df['race'] != 'Caucasian', 0, 1)
compas_df.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)


#Based on class
compas_df_one , compas_df_zero = [x for _, x in compas_df.groupby(compas_df['Probability'] == 0)]

#Based on sex
compas_df_one_female, compas_df_one_male = [x for _, x in compas_df_one.groupby(compas_df_one['sex'] == 0)]
compas_df_zero_female, compas_df_zero_male = [x for _, x in compas_df_zero.groupby(compas_df_zero['sex'] == 0)]


#Based on race
compas_df_one_caucasian, compas_df_one_notcaucasian = [x for _, x in compas_df_one.groupby(compas_df_one['race'] == 0)]
compas_df_zero_caucasian, compas_df_zero_notcaucasian = [x for _, x in compas_df_zero.groupby(compas_df_zero['race'] == 0)]


print(compas_df_one_female.shape, compas_df_one_male.shape, compas_df_zero_female.shape, compas_df_zero_male.shape)
print(compas_df_one_caucasian.shape, compas_df_one_notcaucasian.shape, compas_df_zero_caucasian.shape, compas_df_zero_notcaucasian.shape)

#German dataset visualization
german_df = pd.read_csv('data/GermanData.csv')

german_df['Probability'] = np.where(german_df['Probability'] == 2, 0,1)
german_df['sex'] = np.where(german_df['sex'] == 'A91', 1, german_df['sex'])
german_df['sex'] = np.where(german_df['sex'] == 'A92', 0, german_df['sex'])
german_df['sex'] = np.where(german_df['sex'] == 'A93', 1, german_df['sex'])
german_df['sex'] = np.where(german_df['sex'] == 'A94', 1, german_df['sex'])
german_df['sex'] = np.where(german_df['sex'] == 'A95', 0, german_df['sex'])

#Based on class
german_df_one , german_df_zero = [x for _, x in german_df.groupby(german_df['Probability'] == 0)]

#Based on sex
german_df_one_male, german_df_one_female = [x for _, x in german_df_one.groupby(german_df_one['sex'] == 0)]
german_df_zero_male, german_df_zero_female = [x for _, x in german_df_zero.groupby(german_df_zero['sex'] == 0)]

print(german_df_one_male.shape,german_df_one_female.shape,german_df_zero_male.shape,german_df_zero_female.shape)

#Default dataset visualization

# origin_f = open('data/default_of_credit_card_clients.csv', 'rb')
# new_f = open('data/default_of_credit_card_clients_first_row_removed.csv', 'wb+')
# reader = csv.reader(origin_f)
# writer = csv.writer(new_f)
# for i,row in enumerate(reader):
#     if i > 0:
#        writer.writerow(row)
# origin_f.close()
# new_f.close()

# default_df = pd.read_csv('data/processed.cleveland.data.csv')
# # # default_df = default_df.drop([0])
# # # default_df.columns = ['ID','LIMIT_BAL','sex','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default payment next month']
# default_df['sex'] = np.where(default_df['sex'] == 2, 0,1)
#
# #Based on class
# default_df_one , default_df_zero = [x for _, x in default_df.groupby(default_df['Probability'] == 0)]
#
# #Based on sex
# default_df_one_male, default_df_one_female = [x for _, x in default_df_one.groupby(default_df_one['sex'] == 0)]
# default_df_zero_male, default_df_zero_female = [x for _, x in default_df_zero.groupby(default_df_zero['sex'] == 0)]
#
# print(default_df_one_male.shape,default_df_one_female.shape,default_df_zero_male.shape,default_df_zero_female.shape)

#Heart-Health dataset visualization
heart_df = pd.read_csv('data/processed.cleveland.data.csv')

heart_df['Probability'] = np.where(heart_df['Probability'] > 0, 1, 0)
## calculate mean of age column
mean = heart_df.loc[:,"age"].mean()
heart_df['age'] = np.where(heart_df['age'] >= mean, 0, 1)

# #Based on class
heart_df_one , heart_df_zero = [x for _, x in heart_df.groupby(heart_df['Probability'] == 0)]

#Based on age
heart_df_one_old, heart_df_one_young = [x for _, x in heart_df_one.groupby(heart_df_one['sex'] == 0)]
heart_df_zero_old, heart_df_zero_young = [x for _, x in heart_df_zero.groupby(heart_df_zero['sex'] == 0)]

print(heart_df_one_old.shape,heart_df_one_young.shape,heart_df_zero_old.shape,heart_df_zero_young.shape)

#Bank dataset visualization
from sklearn import preprocessing
bank_df = pd.read_csv('data/bank.csv')

## Drop categorical features

bank_df['Probability'] = np.where(bank_df['Probability'] == 'yes', 1, 0)

mean = bank_df.loc[:,"age"].mean()
bank_df['age'] = np.where(bank_df['age'] >= 30, 1, 0)

#Based on class
bank_df_one , bank_df_zero = [x for _, x in bank_df.groupby(bank_df['Probability'] == 0)]

#Based on age
bank_df_one_old, bank_df_one_young = [x for _, x in bank_df_one.groupby(bank_df_one['age'] == 0)]
bank_df_zero_old, bank_df_zero_young = [x for _, x in bank_df_zero.groupby(bank_df_zero['age'] == 0)]

print(bank_df_one_old.shape,bank_df_one_young.shape,bank_df_zero_old.shape,bank_df_zero_young.shape)

#Student dataset visualization

student_df = pd.read_csv('data/Student.csv')

student_df['sex'] = np.where(student_df['sex'] == 'M', 1, 0)
student_df['Probability'] = np.where(student_df['Probability'] > 12, 1, 0)
#Based on class
student_df_one , student_df_zero = [x for _, x in student_df.groupby(student_df['Probability'] == 0)]


#Based on sex
student_df_one_male, student_df_one_female = [x for _, x in student_df_one.groupby(student_df_one['sex'] == 0)]
student_df_zero_male, student_df_zero_female = [x for _, x in student_df_zero.groupby(student_df_zero['sex'] == 0)]

print(student_df_one_male.shape,student_df_one_female.shape,student_df_zero_male.shape,student_df_zero_female.shape)

dataset_orig = pd.read_csv('data/law_school_clean.csv')
dataset_orig.rename(columns={'male': 'sex'}, inplace=True)
dataset_orig.rename(index=str, columns={'pass_bar': 'Probability'}, inplace=True)
dataset_orig['race'] = np.where(dataset_orig['race'] != 'White', 0, 1)
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 1, 1, 0)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1.0, 1, 0)
#Based on class
student_df_one , student_df_zero = [x for _, x in dataset_orig.groupby(dataset_orig['Probability'] == 0)]


#Based on sex
student_df_one_male, student_df_one_female = [x for _, x in student_df_one.groupby(student_df_one['sex'] == 0)]
student_df_zero_male, student_df_zero_female = [x for _, x in student_df_zero.groupby(student_df_zero['sex'] == 0)]
print("here")
print(student_df_one_male.shape,student_df_one_female.shape,student_df_zero_male.shape,student_df_zero_female.shape)



dataset_orig = pd.read_csv('data/dutch.csv')
dataset_orig.rename(columns={'occupation': 'Probability'}, inplace=True)
mean = dataset_orig.loc[:, "age"].mean()
dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'male', 1, 0)
#Based on class
student_df_one , student_df_zero = [x for _, x in dataset_orig.groupby(dataset_orig['Probability'] == 0)]


#Based on sex
student_df_one_male, student_df_one_female = [x for _, x in student_df_one.groupby(student_df_one['sex'] == 0)]
student_df_zero_male, student_df_zero_female = [x for _, x in student_df_zero.groupby(student_df_zero['sex'] == 0)]
print("here")
print(student_df_one_male.shape,student_df_one_female.shape,student_df_zero_male.shape,student_df_zero_female.shape)

dataset_orig = pd.read_csv('data/default_of_credit_card_clients_first_row_removed.csv')

dataset_orig.rename(index=str, columns={'default payment next month': 'Probability'}, inplace=True)
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0, 1)
mean = dataset_orig.loc[:, "age"].mean()
dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 1, 0)
#Based on class
student_df_one , student_df_zero = [x for _, x in dataset_orig.groupby(dataset_orig['Probability'] == 0)]


#Based on sex
student_df_one_male, student_df_one_female = [x for _, x in student_df_one.groupby(student_df_one['age'] == 0)]
student_df_zero_male, student_df_zero_female = [x for _, x in student_df_zero.groupby(student_df_zero['age'] == 0)]
print("here")
print(student_df_one_male.shape,student_df_one_female.shape,student_df_zero_male.shape,student_df_zero_female.shape)
