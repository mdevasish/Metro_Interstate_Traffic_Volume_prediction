# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:14:03 2020

@author: mdevasish
"""

import pandas as pd
import numpy as np
from mlp.Models import model_construction
import json
import configparser
import sys
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import joblib


def read_dicts():
    '''
    Read the dictionaries from the Disk
    '''
    
    with open('./Data/final_dict.json','r') as fp:
        final_dict = json.load(fp)
    with open('./Data/hour_buckets.json','r') as fp:
        hour_buckets = json.load(fp)
    with open('./Data/month_buckets.json','r') as fp:
        month_buckets = json.load(fp)
    with open('./Data/hol_dict.json','r') as fp:
        hol_dict = json.load(fp)
    return final_dict,hour_buckets,month_buckets,hol_dict

# Extract time based features from the dataframe
def extract_time_feat(df):
    '''
    Extract time based features from the datetime column.
    
    Input Parameters
        
        df : Input Dataframe
        
    Returns
        
        df : Dataframe with datetime features
    '''
    df['hour'] = df['date_time'].apply(lambda x:x.hour)
    df['day'] = df['date_time'].apply(lambda x:x.day)
    df['month'] = df['date_time'].apply(lambda x:x.month)
    df['year'] = df['date_time'].apply(lambda x:x.year)
    df['weekday'] = df['date_time'].apply(lambda x:x.weekday())
    
    # Creating boolean feature to verify if it is weekend
    df.loc[:,'weekend'] = df['weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
    df['weekend'] = df['weekend'].astype('category')
    
    return df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
       'weather_description', 'date_time', 'hour', 'day',
       'month', 'year','weekday','weekend','traffic_volume']]

def feat_cleaning_engg(df,final_dict,hour_buckets,month_buckets,hol_dict):
    '''
    Function to clean the data based on the Exploratory Data Analysis.
    
    Input Parameters
        
        df            : Input data frame that is passed to be cleaned.
        final_dict    : Dictionary related to weather_main and weather_description features needed for feature Engineering
        hour_buckets  : Dictionary related to date_time feature needed for feature engineering
        month_buckets : Dictionary related to date_time feature needed for feature engineering
        
    Output Parameters
    
        df : Cleaned and processed data frame
        
    '''
    df = extract_time_feat(df)
    
    # Imputing the anamoly of temp feature
    df['temp'] = df['temp'].replace(0.0, np.nan)
    df['temp'] = df['temp'].fillna(method = 'ffill')
    df.loc[:,'hour'] = df['hour'].astype(str)
    # Converting the temperature from Kelvin scale to Celsius scale
    df['temp'] = df['temp'] - 273.15
    
    df['holiday'] = df['holiday'].map(hol_dict)
    df['holiday'] = df['holiday'].astype('category')
    
    df['rain_1h'] = df['rain_1h'].replace(df['rain_1h'].max(),np.nan)
    df['rain_1h'] = df['rain_1h'].fillna(df[df['weather_description'] == 'very heavy rain']['rain_1h'].mean())
    
    df['weather_description'] = df['weather_description'].apply(lambda x : x.lower())
    df['weather_description'] = df['weather_description'].map(final_dict)
    
    df['time_slots'] = df['hour'].map(hour_buckets)
    df['time_slots'] = df['time_slots'].astype('category')
    
    df.loc[:,'month'] = df['month'].astype(str)
    df['season'] = df['month'].map(month_buckets)
    df['season'] = df['season'].astype('category')
    
    df = df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
       'weather_description', 'weekend', 'time_slots', 'season', 'traffic_volume']]
    
    
    return df

def preprocessing_lr(df):
    '''
    Function for structuring the data suitable for the ML algorithms
    
    Input parameters :
        
        df : Input Dataframe
    
    Output parameters :
        
        df : Output dataframe
        
    '''
    numeric_feat = ['temp','rain_1h','snow_1h','clouds_all']    
    cat_feat = ['weather_description','time_slots','season']
        
    scaler = MinMaxScaler()
    num_feat = pd.DataFrame(scaler.fit_transform(df[numeric_feat]),columns = numeric_feat)
    joblib.dump(scaler, './Data/scaler.gz') 
    # my_scaler = joblib.load('scaler.gz')
    
    ohe = OneHotEncoder(drop = 'first')
    wd = ohe.fit_transform(df[cat_feat])
    cols = [each[3:] for each in ohe.get_feature_names()]
    ohe_feat = pd.DataFrame(wd.toarray(),columns = cols)
    
    joblib.dump(ohe, './Data/ohe.gz')

    df = df.drop(numeric_feat+['weather_description','time_slots','season'],axis = 1)
    df = pd.concat([num_feat,ohe_feat,df],axis = 1)
    df = df.drop(['clear'],axis = 1)    
    return df

def collect_inputs():
    '''
    Function to read commandline arguments.
    
    Returns :
        tuple of commandline args
    '''
    config = configparser.ConfigParser()
    config.read("config.ini")
    model = str(sys.argv[1])
    
    if model == 'LinearRegression':
        fit_inter = config['LinearRegression']['fit_intercept']
        
        return model,fit_inter,None,None,None
    
    elif model == 'Lasso':
        fit_inter = config['Lasso']['fit_intercept']
        alpha = float(config['Lasso']['alpha'])
        max_iter = int(config['Lasso']['max_iter'])
        
        return model,fit_inter,alpha,max_iter,None
    
    elif model == 'Ridge':
        fit_inter = config['Ridge']['fit_intercept']
        alpha = float(config['Lasso']['alpha'])
        max_iter = int(config['Lasso']['max_iter'])
        solver = config['Ridge']['solver']
        
        return model,fit_inter,alpha,max_iter,solver
    
        

if __name__ == '__main__':
    df = pd.read_csv('./data/traffic.csv',parse_dates = [7])
    final_dict,hour_buckets,month_buckets,hol_dict = read_dicts()
    df = feat_cleaning_engg(df,final_dict,hour_buckets,month_buckets,hol_dict)
    
    new = preprocessing_lr(df)
    model,fit_inter,alpha,max_iter,solver = collect_inputs()
    
    if model == 'LinearRegression':
        print('Evaluation metrics for ' +model)
        z = model_construction(new,model,fit_intercept = fit_inter)
        fimp,diag = z.implement_model(filename = 'LinearRegression')
    
    if model == 'Lasso':
        print('Evaluation metrics for '+model+'Regression')
        z = model_construction(new,model,fit_intercept = fit_inter,alpha = alpha,max_iter = max_iter)
        fimp,diag = z.implement_model(filename = 'Lasso')
        
    if model == 'Ridge':
        print('Evaluation metrics for '+model+'Regression')
        z = model_construction(new,model,fit_intercept = fit_inter,alpha = alpha,max_iter = max_iter,solver = solver)
        fimp,diag = z.implement_model(filename = 'Ridge')
    