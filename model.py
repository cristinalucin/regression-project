import os
import pandas as pd
import numpy as np
import scipy.stats as stats

import env
import wrangle as w

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error
from math import sqrt

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
               columns_to_scale=['bedrooms', 'bathrooms', 'square_feet','home_age'],
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
        
def mvp_model_prep(train,validate,test):
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
    train, validate, test = scale_data(
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
    train, validate, test = scale_data(
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


def model_prep(train,validate,test):
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

def get_mvp_model_with_results(train, validate, test):
    
    '''This function takes in train, validate, test, scales the data using min-max, preps the data for
    modeling, and returns the evaluation of the MVP model via EVS and RMSE'''
    
    #Scale data using min-max
    columns_to_scale = ['bedrooms', 'bathrooms','square_feet']
    # make the object, put it into the variable scaler
    scaler = MinMaxScaler()
    # fit the object to my data:
    train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
    validate[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    #Prep data for modeling iteration 1
    X_train, X_validate, X_test, y_train, y_validate, y_test = mvp_model_prep(train, validate, test)
    #Make a copy of y_train to evaluate results, renaming tax_value
    m1_eval = y_train.copy()
    m1_eval = m1_eval.rename(columns={'tax_value': 'actual'})
    #Establish baseline as mean
    m1_eval['baseline_yhat'] = m1_eval['actual'].mean()
    #Calculate baseline residuals
    m1_eval['residuals'] = m1_eval.baseline_yhat - m1_eval.actual
    #Set model as LR, fit and transform
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train.tax_value)
    #Make Predictions
    m1_eval['ols_yhat'] = lm.predict(X_train)
    #Calculate RMSE
    ols_RMSE = sqrt(mean_squared_error(m1_eval.actual, m1_eval.ols_yhat))
    #Calculate baseline RMSE
    baseline_RMSE = sqrt(mean_squared_error(m1_eval.actual, m1_eval.baseline_yhat))
    # sklearn.metrics.explained_variance_score
    evs = explained_variance_score(m1_eval.actual, m1_eval.ols_yhat)
    
    return print(f'Explained Variance={round(evs,3)}, Baseline Model RMSE: {round(baseline_RMSE,3)}, MVP model RMSE is:{round(ols_RMSE,3)}')


def evaluate_models(X_train, y_train, X_validate, y_validate):
    ''' 
    This function takes in the X and y objects and runs the following models:
    - Baseline model using y_train mean
    - LarsLasso model with alpha=1
    - Polynomial Features 3rd degree with LR
    Then, Returns a DataFrame with the results
    '''
    # Baseline Model
    # run the model
    pred_mean = y_train.tax_value.mean()
    y_train['pred_mean'] = pred_mean
    y_validate['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_validate.tax_value, y_validate.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_mean),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_validate.tax_value, y_validate.pred_mean)}])

    # LassoLars Model
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train.tax_value)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lars, squared=False)
    y_validate['pred_lars'] = lars.predict(X_validate)
    rmse_val = mean_squared_error(y_validate.tax_value, y_validate.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_lars),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_validate.tax_value, y_validate.pred_lars)}, ignore_index=True)

    # Polynomial Model, 3rd Degree
    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d3 = pf.fit_transform(X_train)
    X_val_d3 = pf.transform(X_validate)
    
    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d3, y_train.tax_value)
    y_train['pred_lm2'] = lm2.predict(X_train_d3)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm2, squared=False)
    y_validate['pred_lm2'] = lm2.predict(X_val_d3)
    rmse_val = mean_squared_error(y_validate.tax_value, y_validate.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'Polynomial Features 3rd Degree',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_validate.tax_value, y_validate.pred_lm2)}, ignore_index=True)

    return metrics

def evaluate_model_test(X_train, X_validate, X_test, y_train, y_validate, y_test):
    ''' 
    This function takes in the X and y objects and then runs Polynomial Features 3rd
    Degree Regression Model. It then returns the results in a dataframe, including model
    performance on testing data
    '''
    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d3 = pf.fit_transform(X_train)
    X_val_d3 = pf.transform(X_validate)
    X_test_d3 = pf.transform(X_test)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d3, y_train.tax_value)
    y_train['pred_lm2'] = lm2.predict(X_train_d3)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm2, squared=False)
    y_validate['pred_lm2'] = lm2.predict(X_val_d3)
    rmse_val = mean_squared_error(y_validate.tax_value, y_validate.pred_lm2, squared=False)
    y_test['pred_lm2'] = lm2.predict(X_test_d3)
    rmse_test = mean_squared_error(y_test.tax_value, y_test.pred_lm2, squared=False)

    # save the results
    results = pd.DataFrame({'train': 
                               {'rmse': rmse_train, 
                                'r2': explained_variance_score(y_train.tax_value, y_train.pred_lm2)},
                           'validate': 
                               {'rmse': rmse_val, 
                                'r2': explained_variance_score(y_validate.tax_value, y_validate.pred_lm2)},
                           'test': 
                               {'rmse': rmse_test, 
                                'r2': explained_variance_score(y_test.tax_value, y_test.pred_lm2)}
                          })
    
    return results