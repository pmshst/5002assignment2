#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cai Zhao
# python version 3.6.5
# using pep8 pycodestyle
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
df = pd.read_csv("trainFeatures.csv")
mode_workclass = df['workclass'].mode()[0]
mode_occupation = df['occupation'].mode()[0]
mode_native_country = df['native-country'].mode()[0]
df['workclass'].replace(' ?', mode_workclass, inplace=True)
df['occupation'].replace(' ?', mode_occupation, inplace=True)
df['native-country'].replace(' ?', mode_native_country, inplace=True)
df = pd.get_dummies(df)
label = pd.read_csv('trainLabels.csv', header=None)
df['more_than_50k'] = label
y = df['more_than_50k']
del df['more_than_50k']
features = df.columns[:len(df.columns)]
clf = GradientBoostingClassifier(
    n_estimators=1750,
    learning_rate=0.6000000000000001,
    max_depth=1,
    random_state=0)
clf.fit(df, y)
test_df = pd.read_csv("testFeatures.csv")
mode_workclass = test_df['workclass'].mode()[0]
mode_occupation = test_df['occupation'].mode()[0]
mode_native_country = test_df['native-country'].mode()[0]
test_df['workclass'].replace(' ?', mode_workclass, inplace=True)
test_df['occupation'].replace(' ?', mode_occupation, inplace=True)
test_df['native-country'].replace(' ?', mode_native_country, inplace=True)
test_df = pd.get_dummies(test_df)
test_features = test_df.columns[:len(test_df.columns)]
for i in features:
    if i not in test_features:
        test_df[i] = 0
test_preds = clf.predict(test_df)
file_o = open('prediction.csv', 'w')
for i in test_preds:
    file_o.write(str(i) + '\n')
file_o.close()
