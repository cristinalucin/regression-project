import os
import pandas as pd
import numpy as np
import scipy.stats as stats

import env
import wrangle as w

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 15% of the original dataset, validate is .1765*.85= 15% of the 
    original dataset, and train is 75% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.15, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.1765, 
                                       random_state=seed)
    return train, validate, test


def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    '''This function visualizes the difference in original data and scaled data. It takes the scaler type,
    dataframe, columns included in scaling, and outputs multiple visualizations based on those
    features. The 
    '''
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
#    return fig, axs

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'area','tax_value'],
               return_scaler=False):
    '''This function takes in train, validate, test, and outputs scaled data based on
    the chosen method (quantile scaling) using the columns selected as the only columns
    that will be scaled. This function also returns the scaler object as an array if set 
    to true'''
    # make copies of our original data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
     # select a scaler
    scaler = QuantileTransformer()
     # fit on train
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
def get_dummies(train,validate,test):
    '''This function generates dummy variables for county and renames counties appropriately'''
    
    #Create dummies for county
    train = pd.get_dummies(train, columns=['county'], drop_first=False)
    validate = pd.get_dummies(validate, columns=['county'], drop_first=False)
    test = pd.get_dummies(test, columns=['county'], drop_first=False)
    
    return train, validate, test
        
def model1_prep(train,validate,test):
    '''
    This function prepares train, validate, test for model 1 by dropping columns not necessary
    or compatible with modeling algorithms.
    '''
    # drop columns not needed for model 1
    keep_cols = ['bedrooms',
                 'bathrooms',
                 'square_feet',
                 'tax_value',
                 ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]

    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    X_train = train.drop(columns='tax_value').reset_index(drop=True)
    y_train = train[['tax_value']].reset_index(drop=True)

    X_validate = validate.drop(columns='tax_value').reset_index(drop=True)
    y_validate = validate[['tax_value']].reset_index(drop=True)

    X_test = test.drop(columns='tax_value').reset_index(drop=True)
    y_test = test[['tax_value']].reset_index(drop=True)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def get_clean_data():
    '''This function retrieves clean data to perform modeling'''
    #Retrieve Zillow data from CSV or Codeup mySQL database
    df = w.get_zillow_data()
    #Clean the data
    df = w.clean_zillow(df)
    #Split the data
    train, validate, test = w.train_validate_test_split(df)
    
    return train, validate, test

def model2_prep(train,validate,test):
    '''
    This function prepares train, validate, test for model 2 by dropping columns not necessary
    or compatible with modeling algorithms, splitting data into target and feature (X and Y), and
    scaling data for modeling
    '''
    #Scaling Data
    train, validate, test = m.scale_data(
        train, validate, test, columns_to_scale=['bathrooms', 'square_feet','home_age'], return_scaler=False)
    
    #Make Dummy Variables for county
    train = pd.get_dummies(train, columns=['county'], drop_first=False)
    validate = pd.get_dummies(validate, columns=['county'], drop_first=False)
    test = pd.get_dummies(test, columns=['county'], drop_first=False)
    
    #Change column names for dummy variables
    train = train.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    validate = validate.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    test = test.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    
    # drop columns not needed for model 2
    keep_cols = ['bathrooms',
                 'square_feet',
                 'home_age',
                 'LA',
                 'tax_value',
                 ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]

    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    X_train = train.drop(columns='tax_value').reset_index(drop=True)
    y_train = train[['tax_value']].reset_index(drop=True)

    X_validate = validate.drop(columns='tax_value').reset_index(drop=True)
    y_validate = validate[['tax_value']].reset_index(drop=True)

    X_test = test.drop(columns='tax_value').reset_index(drop=True)
    y_test = test[['tax_value']].reset_index(drop=True)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def model3_prep(train,validate,test):
    '''
    This function prepares train, validate, test for model 3 by dropping columns not necessary
    or compatible with modeling algorithms, splitting data into target and feature (X and Y), and
    scaling data for modeling
    '''
    #Scaling Data
    train, validate, test = m.scale_data(
        train, validate, test, columns_to_scale=['bedrooms', 'square_feet','home_age'], return_scaler=False)

    #Make Dummy Variables for county
    train = pd.get_dummies(train, columns=['county'], drop_first=False)
    validate = pd.get_dummies(validate, columns=['county'], drop_first=False)
    test = pd.get_dummies(test, columns=['county'], drop_first=False)
    
    #Change column names for dummy variables
    train = train.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    validate = validate.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    test = test.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    
    # drop columns not needed for model 3
    keep_cols = ['bedrooms',
                 'square_feet',
                 'home_age',
                 'LA',
                 'tax_value',
                 ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]

    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    X_train = train.drop(columns='tax_value').reset_index(drop=True)
    y_train = train[['tax_value']].reset_index(drop=True)

    X_validate = validate.drop(columns='tax_value').reset_index(drop=True)
    y_validate = validate[['tax_value']].reset_index(drop=True)

    X_test = test.drop(columns='tax_value').reset_index(drop=True)
    y_test = test[['tax_value']].reset_index(drop=True)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def model4_prep(train,validate,test):
    '''
    This function prepares train, validate, test for model 4 by dropping columns not necessary
    or compatible with modeling algorithms, splitting data into target and feature (X and Y), and
    scaling data for modeling (using quantile transformer)
    '''
    #Scaling Data
    train, validate, test = scale_data(
        train, validate, test, columns_to_scale=['bedrooms','bathrooms','square_feet','home_age'], return_scaler=False)

    #Make Dummy Variables for county
    train = pd.get_dummies(train, columns=['county'], drop_first=False)
    validate = pd.get_dummies(validate, columns=['county'], drop_first=False)
    test = pd.get_dummies(test, columns=['county'], drop_first=False)
    
    #Change column names for dummy variables
    train = train.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    validate = validate.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    test = test.rename(columns={'county_Los Angeles': 'LA', 'county_Orange':'Orange','county_Ventura':'Ventura'})
    
    # drop columns not needed for model 4
    keep_cols = ['bedrooms',
                 'bathrooms',
                 'square_feet',
                 'home_age',
                 'LA',
                 'tax_value',
                 ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]

    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    X_train = train.drop(columns='tax_value').reset_index(drop=True)
    y_train = train[['tax_value']].reset_index(drop=True)

    X_validate = validate.drop(columns='tax_value').reset_index(drop=True)
    y_validate = validate[['tax_value']].reset_index(drop=True)

    X_test = test.drop(columns='tax_value').reset_index(drop=True)
    y_test = test[['tax_value']].reset_index(drop=True)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test