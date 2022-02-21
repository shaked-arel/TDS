#data processing
from ast import Mod
import csv
import pandas as pd
import numpy as np
import scipy as sp

#statistics
from scipy import stats
import statsmodels.api as sm

#data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

#Machine learning library
import sklearn

#Patterns Mining
from efficient_apriori import apriori

import warnings
warnings.filterwarnings("ignore")

'''choose a dataset'''
#dataset_path = "data/car_data_nan.csv"
#dataset_path = "data/balance_nan.csv"
#dataset_path = "data/ta_nan.csv"
dataset_path = "data/shopping_data_nan.csv"
dtf = pd.read_csv(dataset_path)


categorical_columns = [c for c in dtf.columns]
column_name = []
nan_rows = []

'''find all the columns that have a nan value in one of their attributes'''
na_columns = dtf[categorical_columns].isna().sum()
na_columns = na_columns[na_columns>0]

'''find the index of a row with nan value'''
for nc in na_columns.index:
    column_name.append(nc)
    mask = dtf[nc].isna()
    for i in dtf[mask].index:
        if i not in nan_rows:
            nan_rows.append(i)
    

# data frame to dictionary
df=dtf.copy()
records = df.to_dict(orient='records')

# create list of transacttions
transactions=[]
for r in records:
    transactions.append(list(r.items()))

# find the rules by apriory
itemsets, rules = apriori(transactions, min_support=0.05, min_confidence=0.5,output_transaction_ids=False)

# algorithm
'''go over all the rows that have a nan value'''
for row_index in nan_rows:
    left_side = []
    right_side = []
    row = df.iloc[[row_index]] #find the row by its index
    '''for each row we find the categories of the nan values and the rest of the categories that have values'''
    for column in column_name:
        check = row[column].isna()
        '''nan categories'''
        if check.values:
            right_side.append(column)
    for column in categorical_columns:
        '''not nan categories and their values'''
        if column not in right_side:
            c = row[column].values[0]
            left_side.append((column,row[column].values[0]))
    '''find relevant rules for each row'''
    part_rules = []
    '''rules that the right side of the rule containes the right side of the row'''
    for r in rules:
        for right in right_side:
            for i in r.rhs:
                c = i[0]
                if right == i[0]:
                    if r not in part_rules:
                        part_rules.append(r)
    relevant_rules = []
    '''rules that the left side of the row containes the left side of the rule'''
    for r in part_rules:
        for left in r.lhs:
            if left in left_side:
                if r not in relevant_rules:
                    relevant_rules.append(r)
    '''for each nan value go over the relevant rules and fill in the missing value'''
    for right in right_side:
        rules_values = []
        for r in relevant_rules:
            for category in r.rhs: 
                #check if the category we want to fill fits with the right side of the rule
                if category[0]==right:
                    # save the value of the right side of the rule
                    rules_values.append(category[1])
        #check if we found any rules
        if len(rules_values)!=0:
            #insert instead of the nan the most frequent value
            mode = (max(set(rules_values), key = rules_values.count))
            dtf.loc[row_index, right] = mode
#update the csv file
dtf.to_csv("shopping_filled2.csv", index=False) 
