# Metro_Interstate_Traffic_Volume_prediction
Predict the hourly traffic volume of Minneapolis-St Paul Interstate I-94 based on weather conditions and list of holidays


Author Name : PATNANA MANIKANTHA DEVASISH

Problem Statement :
The objective of the project is to predict the approximate traffic volume for a specific point of date and time, given the climatic conditions like rainfall, temperature, percentage of the cloud cover, snowfall and the textual description of climate.

Assumptions :

1. As the task is to predict the hourly traffic volumes, practically such problem statements qualify for a forecast, which takes seasonality and trend into account. But we assume that there is neither upward nor downward trend in the dataset.

Metrics used :
The metrics monitored here are root mean squared error, mean absolute error and R square.

Overview of the Deliverable Folder and Folder Structure :

data 				: Folder containing the csv files and metadata related to MinMaxscaler and onehotencoder etc.
mlp  				: Folder containing the implementation of ML pipeline named Models.py
Models 				: Folder contain the persisted version of the model
Screenshots 		        : Screenshot of a plot used in EDA.ipynb
.ipynb_checkpoints              : Checkpoints of the python Notebooks

Folder Structure :

|	.ipynb_checkpoints

		|-> eda-checkpoint.ipynb
		|-> Modelling-checkpoint.ipynb
|	data

		|-> final_dict.json	
		|-> hol_dict.json
		|-> hour_buckets.json
		|-> month_buckets.json
		|-> ohe.gz
		|-> processed.csv
		|-> scaler.gz
		|-> le.gz
		|-> traffic.csv
|	mlp

		|-> __pycache__
				|-> Models.cpython-37.pyc
				|-> Models.cpython-38.pyc
		|-> Models.py
|	Models

		|-> LassoRegression.sav
		|-> LinearRegression.sav
		|-> RidgeRegression.sav
|	Screenshots

		|-> snow.jpg
|->	config.ini
|-> eda.ipynb
|-> Extraction.py
|-> Modelling.ipynb
|-> README.md
|-> requirements.txt
|-> run.py
|-> run.sh

Basic Insights from EDA :

Below are the basic insights and the actions to be taken on the data to clean and structure the data for modelling for each feature present in the data. 

i) holiday : 

Based on the distribution of the traffic volume during the public holidays, New Years day have greater traffic volume compared to the rest of the other holidays. A logical grouping of these holidays in to various buckets based on the traffic pattern will make things easy for the modelling.

ii) temp : 

Temperature feature has an anamoly value around 0. Such observation can be a reading from a faulty sensor.This has to be imputed based on the nearest neighbours(based on the time of the record).

iii) rain_1h :

Rain feature has an anamoly value of 9831.3 . This value is recorded as 'very heavy rain' under weather description. This observation has to be imputed based on the average rainfall for 'very heavy rain' category under weather description.


iv) snow_1h :

Based on the univariate analysis of snow_1h, the feature has large number of records near 0. And the plot suggests that there are few outliers, but practically these values look to be real enough.

v) clouds_all :

This feature has has two big humps around 0 & 90 and small hump around 40.

vi) Weather_main & Weather_description :

weather_main and weather_description describe the weather condition textually. The descriptions include rainy, cloudy, smoke, clear, fog, haze, drizzle, mist and squall. These features can be clubbed together and form a single feature.

vii) date_time :

The timeline of the database ranges from '2013-01-01 00:00:00' to '2018-09-30 23:00:00'. The observations are recorded every hour of each day. There are few inconsistencies in the timeline of the data. Records for a period of 6 months are not collected. The time period the data was not collected is in the year 2015 from Jan to Jun.

viii) traffic_volume :

The readings of the traffic volume are not recorded continuously. Traffic volume records are missing in the year 2014 in the month of May and in the year 2015 for a period of 6 months(Jan-Jun) or more. There are few records that have multiple weather descriptions of the weather feature to describe the traffic volume for specific point of time.

Pipeline Design :

Extract Data from the Database (File : Extraction.py) -> Raw Data -> Extract Date time Features (Name of the function : extract_time_feat) -> Data Cleaning and Feature Engineering (Name of the function : feat_cleaning_engg) -> Data preprocessing (Name of the function : preprocessing_lr) -> Implementation of ML Algorithms (Name of the class : model_construction)

Steps involved in the constructing the ML pipeline are as follows :

1. Data is extracted from the database and saved in the form of csv file.
2. Perform Exploratory Data Analysis
3. Read raw data from the csv file and extract date time features.
4. Further clean the data and perform feature engineering based on the insights from the Exploratory Data Analysis. 
5. Structure the data based on the ML algorithm that is implemented.
6. Implement the ML algorithm, fine tune the hyperparameters for desired results and monitor the evaluation results.

Models used :

Models implemented are Linear regression, Lasso Regression, Ridge Regression and DecisionTreeRegressor. The rationale behind the choise of models is as follows

1. Linear Regression along with the regularization variants Lasso Regression and Ridge regression were primarily chosen for the implementation because they form the baseline family of regression suite.

2. Secondly, this suite of models have the ability to make the stakeholders explain the model in the form of an equation. Business stakeholders also get to understand the impact of slight change in the input features.

3. Equation based algorithms have been incapable in predicting the traffic volume during the wee hours. And for the sole reason, DecisionTreeRegressor is used.

Instructions to use the bash file :

The bash file is located at the base folder.

Modify the contents of shell script to implement LinearRegression : python run.py LinearRegression
Modify the contents of shell script to implement Lasso Regression : python run.py Lasso
Modify the contents of shell script to implement Ridge Regression : python run.py Ridge

The hyperparameters of the models are extracted from 'config.ini' file. Any changes in the hyperparameters can be modified through the config file.

Results :

1. All the implemented models showed no signs of either overfitting or underfitting.
2. The metrics root mean squared error and R square have been consistent across all the algorithms despite implementing regularisation based algorithms with highest regularisation. Hyperparameter tuning is not conducted here because the models did not show any signs of either overfitting or underfitting.
3. The diagnostic plots in all the models reveal that the models have been resilient in predicting the high traffic volumes and failed mostly in predicting the low traffic volumes for equation based models. 
4. All the four models have predicted negative traffic volume, which is not acceptable. But out of all the equation based models, Lasso Regression has performed better in terms of number of predictions with low negative traffic volume. And the DecisionTreeRegressor has the best performance. 
5. Based on the diagnostics plots, the feature engineering for low traffic volumes have to be more strong to reduce the prediction error.
6. In order to offset the negative traffic volume, few advanced models like DecisionTreeRegressor are be implemented and improve the prediction capability of the models.

