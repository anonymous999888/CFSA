import os
import math
import argparse
import copy
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import operator
from Geneate_data import generate_data

def bias_point(data, clf):
    prob1 = clf.predict_proba(data)
    if prob1 >= 0.5:
        res1 = 1
    else:
        res1 = 0
    if data[0] == 1:
        data[0] = 0
    else:
        data[0] = 1
    prob2 = clf.predict_proba(data.iloc[:,0:15])
    if prob2 >= 0.5:
        res2 = 1
    else:
        res2 = 0
    if res1 == res2:
        return False
    else:
        return True

def remove_bias_point(dataset_orig_test,protected_attribute,dataset_used):
    if dataset_used == 'bank':
        zero_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
    else:
        zero_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])
    a = zero_one + one_one
    b = -1 * (zero_zero * zero_one + 2 * zero_zero * one_one + one_zero * one_one)
    c = (zero_zero + one_zero) * (zero_zero * one_one - zero_one * one_zero)
    x = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    y = (zero_one + one_one) / (zero_zero + one_zero) * x

    zero_zero_new = int(zero_zero - x)
    one_one_new = int(one_one - y)
    zero_zero_diff = zero_zero - zero_zero_new
    one_one_diff = one_one - one_one_new
    if dataset_used == 'bank':
        zero_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)]
        one_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)]
        zero_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)]
    else:
        zero_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)]
        one_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)]
        zero_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)]
    return one_one_diff, zero_zero_diff, one_one_set.index, zero_zero_set.index, zero_one_set.index, one_zero_set.index


def transfer_sen_att(dataset_orig_train, attr):
    for index, row in dataset_orig_train.iterrows():
        if row[attr] == 0:
            dataset_orig_train.loc[index, attr] = 1
        else:
            dataset_orig_train.loc[index, attr] = 0
    return dataset_orig_train

def compare_res(pred1,pred2, index, one_one_set_index, zero_zero_set_index, one_one_diff, zero_zero_diff, zero_one_set_index, one_zero_set_index, dataset_used):
    res1_diff = []
    res2_diff = []
    res1 = []
    for i in range(len(pred1)):
        prob_t = (pred1[i][1])
        res1_diff.append(prob_t - 0.5)
        if prob_t >= 0.5:
            res1.append(1)
        else:
            res1.append(0)
    res2 = []
    for i in range(len(pred2)):
        prob_t = (pred2[i][1])
        res2_diff.append(prob_t - 0.5)
        if prob_t >= 0.5:
            res2.append(1)
        else:
            res2.append(0)
    label_diff = []
    for i in range(len(res1)):
        if res1[i] != res2[i]:
            label_diff.append(index[i])
    # print(label_diff)
    one_one_prob_diff = {}
    zero_zero_prob_diff = {}
    nonchange_one_one_prob_diff = {}
    nonchange_zero_zero_prob_diff = {}
    print("label diff", len(label_diff))
    #gen data two group
    zero_one_prob_diff = {}
    one_zero_prob_diff = {}
    nonchange_zero_one_prob_diff = {}
    nonchange_one_zero_prob_diff = {}

    for i in range(len(res1_diff)):
        diff = res1_diff[i] - res2_diff[i]
        if index[i] in label_diff and index[i] in one_one_set_index:
            one_one_prob_diff[index[i]] = abs(diff)
        if index[i] in label_diff and index[i] in zero_zero_set_index:
            zero_zero_prob_diff[index[i]] = abs(diff)
        if index[i] in label_diff and index[i] in zero_one_set_index:
            zero_one_prob_diff[index[i]] = abs(diff)
        if index[i] in label_diff and index[i] in one_zero_set_index:
            one_zero_prob_diff[index[i]] = abs(diff)
        if index[i] not in label_diff and index[i] in one_one_set_index:
            nonchange_one_one_prob_diff[index[i]] = abs(diff)
        if index[i] not in label_diff and index[i] in zero_zero_set_index:
            nonchange_zero_zero_prob_diff[index[i]] = abs(diff)
        if index[i] not in label_diff and index[i] in zero_one_set_index:
            nonchange_zero_one_prob_diff[index[i]] = abs(diff)
        if index[i] not in label_diff and index[i] in one_zero_set_index:
            nonchange_one_zero_prob_diff[index[i]] = abs(diff)

    one_one_prob_diff = dict(sorted(one_one_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    zero_zero_prob_diff = dict(sorted(zero_zero_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    zero_one_prob_diff = dict(sorted(zero_one_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    one_zero_prob_diff = dict(sorted(one_zero_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    nonchange_one_one_prob_diff = dict(sorted(nonchange_one_one_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    nonchange_zero_zero_prob_diff = dict(sorted(nonchange_zero_zero_prob_diff.items(), key=operator.itemgetter(1), reverse=True))
    nonchange_zero_one_prob_diff = dict(sorted(nonchange_zero_one_prob_diff.items(), key=operator.itemgetter(1), reverse=False))
    nonchange_one_zero_prob_diff = dict(sorted(nonchange_one_zero_prob_diff.items(), key=operator.itemgetter(1), reverse=False))
    one_one_ranklist = []
    zero_zero_ranklist = []
    zero_one_ranklist = []
    one_zero_ranklist = []
    zero_one_creatlist = []
    one_zero_creatlist = []
    flipping_list=[]
    for i in one_one_prob_diff.keys():
        one_one_ranklist.append(i)
    for i in zero_zero_prob_diff.keys():
        zero_zero_ranklist.append(i)
        flipping_list.append(i)
    for i in zero_one_prob_diff.keys():
        zero_one_ranklist.append(i)
    for i in one_zero_prob_diff.keys():
        one_zero_ranklist.append(i)
    for i in nonchange_zero_one_prob_diff.keys():
        zero_one_creatlist.append(i)
    for i in nonchange_one_zero_prob_diff.keys():
        one_zero_creatlist.append(i)
    # if len(one_one_ranklist) < one_one_diff:
    for i in nonchange_one_one_prob_diff:
        one_one_ranklist.append(i)
    # if len(zero_zero_ranklist) < zero_zero_diff:
    for i in nonchange_zero_zero_prob_diff.keys():
        zero_zero_ranklist.append(i)

    return one_one_ranklist, zero_zero_ranklist, zero_one_ranklist, one_zero_ranklist, zero_one_creatlist, one_zero_creatlist,flipping_list


def data_biased_dis(dataset_orig_train, attr, dataset_used):
    one_one_diff, zero_zero_diff, one_one_set_index, zero_zero_set_index, zero_one_set_index, one_zero_set_index = remove_bias_point(dataset_orig_train, attr, dataset_used)
    # X_unbias, y_unbias = dataset_orig_train_unbias.loc[:, dataset_orig_train_unbias.columns != 'Probability'], dataset_orig_train_unbias['Probability']
    X, y = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    # print(dataset_orig_train)
    # print(dataset_orig_train["sex"])
    for index, row in dataset_orig_train.iterrows():
        if row[attr] == 0:
            dataset_orig_train.loc[index, attr] = 1
        else:
            dataset_orig_train.loc[index, attr] = 0
    # print("mark1",dataset_orig_train)
    # print(dataset_orig_train["sex"])
    X_unbias, y_unbias = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf.fit(X, y)
    pred1 = clf.predict_proba(X)
    pred2 = clf.predict_proba(X_unbias)
    print(pred1,pred2)
    print(len(pred1), len(pred2))
    one_one_ranklist, zero_zero_ranklist, zero_one_ranklist, one_zero_ranklist, zero_one_creatlist, one_zero_creatlist, flipping_list = compare_res(pred1, pred2, X.index, one_one_set_index, zero_zero_set_index, one_one_diff, zero_zero_diff, zero_one_set_index, one_zero_set_index, dataset_used)
    if len(one_one_ranklist) == 0:
        print("one_one_ranklist error")
        return False
    if len(zero_zero_ranklist) == 0:
        print("zero_zero_ranklist error")
        return False
    for index, row in dataset_orig_train.iterrows():
        if row[attr] == 0:
            dataset_orig_train.loc[index, attr] = 1
        else:
            dataset_orig_train.loc[index, attr] = 0
    flipping_sample = dataset_orig_train.copy(deep=True)
    if dataset_used == 'bank':
        for index, row in flipping_sample.iterrows():
            if row[attr] == 0 and row['Probability'] == 1:
                flipping_sample.loc[index, 'Probability'] = 0
        flipping_data = flipping_sample[(flipping_sample['Probability'] == 0) & (flipping_sample[attr] == 0)]
    else:
        for index, row in flipping_sample.iterrows():
            if row[attr] == 0 and row['Probability'] == 0:
                flipping_sample.loc[index, 'Probability'] = 1
        flipping_data = flipping_sample[(flipping_sample['Probability'] == 1) & (flipping_sample[attr] == 0)]
    flipping_drop = []
    for i in flipping_data.index:
        if i not in flipping_list:
            flipping_drop.append(i)
    flipping_data = flipping_data.drop(flipping_drop)
    c = 0
    for index, row in dataset_orig_train.iterrows():
        if row[attr] == 0 and row['Probability'] == 1:
            c += 1
    if dataset_used == 'bank':
        zero_one_set = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 1)]
        one_zero_set = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 0)]
        zero_zero_set = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 0)]
        one_one_set = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 1)]
    else:
        zero_one_set = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 1)]
        one_zero_set = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 0)]
        zero_zero_set = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 0)]
        one_one_set = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 1)]
    newset = zero_one_set.append(one_zero_set)
    for i in zero_zero_set.index:
        if i not in zero_zero_ranklist[:zero_zero_diff]:
            # print(dataset_orig_train[(dataset_orig_train.index == i)])
            newset = newset.append(dataset_orig_train[(dataset_orig_train.index == i)])
    for i in one_one_set.index:
        if i not in one_one_ranklist[:one_one_diff]:
            newset = newset.append(dataset_orig_train[(dataset_orig_train.index == i)])



    droplist = zero_one_ranklist + one_zero_ranklist


    newset = newset.drop(droplist)

    newset = newset.append(flipping_data)

    create_list = zero_one_creatlist+one_zero_creatlist
    l1 = len(one_zero_ranklist) - len(flipping_list)
    print("l1 fix",l1)
    print("length of zero_one_creatlist",len(zero_one_creatlist))
    newset = generate_data(newset, len(droplist), len(zero_one_ranklist), l1, zero_one_creatlist, one_zero_creatlist, dataset_used)
    print(newset)
    newset['Probability'] = np.where(newset['Probability'] >= 0.5, 1, 0)
 
    print(newset)

    return newset
