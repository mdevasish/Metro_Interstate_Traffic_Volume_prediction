# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:14:16 2020

@author: mdevasish
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


class model_construction():
    
    def __init__(self,data,model,fit_intercept = True,alpha = 1.0, max_iter = 1000, solver = 'auto'):
        '''
        Constructor to set the values before creating the model
        
        Input Parameters :
        
            data     : Input DataFrame
            model    : Model to be implemented
            alpha    : Regularization constant applicable for Ridge and Lasso
            max_iter : Maximimum iterations applicable for Lasso
            solver   : Type of solver to use applicable for Ridge
        
        '''
        self.data = data
        self.alpha = alpha
        self.max_iter = max_iter
        self.solver = solver
        self.fit_intercept = fit_intercept
        if model == 'LinearRegression':
            self.model = LinearRegression(fit_intercept = self.fit_intercept)
        elif model == 'Lasso':
            self.model = Lasso(alpha = self.alpha,max_iter = self.max_iter,fit_intercept = self.fit_intercept)
        elif model == 'Ridge':
            self.model = Ridge(alpha = self.alpha,solver = self.solver,fit_intercept = self.fit_intercept)
        else:
            raise Exception('Wrong input model')
    
    def implement_model(self,filename):
        '''
        Method inside the model_construction class, used for implementing the model
        and return feature importance and dataframe with actual values and predicted values of validation set
        
        Input :
            tsize      : size of the dataset for the validation default value 0.3
            random_val : Seed for randomness for reproducibility default value 2020
        
        Returns :
            fimp : Feature importance of a model
            diag : diagnostic dataframe with actual values and predicted values of validation set
        '''
        df = self.data
        model = self.model
        
        
        X,y = df.iloc[:,:-1],df.iloc[:,-1]
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3,random_state = 2020)
        
        model.fit(X_train,y_train)
        
        print('R square score on train set and test set are :',model.score(X_train,y_train),model.score(X_val,y_val))
        print('Root mean squared error on test set is :',np.sqrt(mean_squared_error(y_val,model.predict(X_val))))
        print('Mean absolute error on test set is :',mean_absolute_error(y_val,model.predict(X_val)))
        
        fimp = pd.DataFrame(zip(X.columns,model.coef_),columns = ['feat','coeff']).sort_values(by = 'coeff',ascending = False)
        fimp['abs_coeff'] = fimp['coeff'].apply(lambda x : x if x > 0 else -x)
        fimp['rel'] = fimp['coeff'].apply(lambda x : 'pos' if x > 0 else 'neg')
        fimp['rel'] = fimp['rel'].astype('category')
        fimp = fimp.sort_values(by = 'abs_coeff',ascending = False)
        
        pred = model.predict(X_val)
        diag = pd.DataFrame(zip(y_val,pred),columns = ['Ground Truth','Predicted'])
        
        full_name = './Models/'+filename+'.sav'
        joblib.dump(model, full_name)
        return fimp,diag

    def plot_feat_imp(self,fimp,title):
        '''
        Method inside the model_construction class, used for creating a feature importance plot
        
        Input :
            fimp  : Dataframe with feature importance
            title : Title of the plot
        
        Displays a plot
        '''
        plt.figure(figsize = (18,12))
        sns.barplot(y = 'feat', x = 'abs_coeff', hue = 'rel',data = fimp)
        plt.title('Feature Importance plot for '+title)
        
    def plot_diagnostic(self,diag):
        '''
        Method inside the model_construction class, used for creating a diagnostic plot ground truth vs predicted
        
        Input :
            diag  : Dataframe with feature importance
            
        Displays a plot
        '''
        
        plt.figure(figsize = (18,9))
        g = sns.scatterplot(x = 'Ground Truth', y = 'Predicted',data = diag)
        plt.title('Ground Truth vs Predicted on validation Data')
        plt.show()
