import pandas as pd
import numpy as np


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def d_type(x):
    types = []
    for xi in x:
        types.append(train[xi].dtype)
    return types

def null_counts(data):
    #return number of missing values and proportion
    missing = data.isnull().sum().sort_values(ascending=False)
    percent = missing/len(data)
    frame = pd.concat([missing,percent],axis=1,keys=['missing','prop'])
    frame['type'] = d_type(list(frame.index))
    
    return frame.reset_index()
def drop_cols(train,test):
    #drop all columns in train set with missing values > 81.
    #drops same columns in test set
    nulls = null_counts(train)
    droplist = list(nulls['index'][nulls.missing > 81])
    train = train.drop(droplist,1)
    test = test.drop(droplist,1)
    return train,test

def get_numeric_missing(df):
    #returns list of numeric type column names containnig missing values
    df_nulls = null_counts(df)
    numerics = df_nulls['index'][(df_nulls['missing']>0)&
                                   ((df_nulls['type']=='float64')| (df_nulls['type']=='int64'))].to_list()
    return numerics

def get_cats_missing(df):
    #returns list of object type columns containing missing values
    df_nulls = null_counts(df)
    cats = df_nulls['index'][(df_nulls['missing']>0) & (df_nulls['type'] == 'object')].to_list()
    return cats

def fill_nulls(train,test):
    #nulls for these columns means the item does not exist.
    nonexistant = ['GarageType','GarageFinish','GarageCond','GarageQual',
            'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual']
    
    train[nonexistant]=train[nonexistant].fillna('nil')
    test[nonexistant]=test[nonexistant].fillna('nil')
    

    numeric_train = get_numeric_missing(train)
    cats_train = get_cats_missing(train)
    
    train[numeric_train] = train[numeric_train].fillna(train[numeric_train].apply(np.mean))
    train[cats_train] = train[cats_train].fillna(train[cats_train].mode().iloc[0,:])
    
    
    numeric_test = get_numeric_missing(test)
    cats_test = get_cats_missing(test)
    
    test[cats_test] = test[cats_test].fillna(train[cats_test].mode().iloc[0,:])
    test[numeric_test] = test[numeric_test].fillna(test[numeric_test].apply(np.mean))
    
    return train,test

train,test = drop_cols(train,test)
train,test = fill_nulls(train,test)





