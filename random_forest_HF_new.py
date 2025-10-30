import numpy as np
# manipulating data via DataFrames, 2-D tabular, column-oriented data structure
import pandas as pd
from pandas import read_excel
from sklearn.manifold import TSNE
my_sheet = 'FinalAnalysis_forManuscript'
file_name = 'Data_for_ML_crop_residues_assessment.xlsx'
from math import sqrt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# producing plots and other 2D data visualizations. Use plotly if you want interactive graphs
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
# statistical visualizations (a wrapper around Matplotlib)
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
from sklearn.metrics import classification_report,confusion_matrix
# from deepforest import CascadeForestRegressor

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.svm import SVR
import joblib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor
from sklearn.metrics import PredictionErrorDisplay

import os
import scipy
from scipy.stats import skew,kurtosis

os.makedirs('./subfolder', exist_ok=True)


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

# Assign the csv data to a DataFrame
# inputExcelFile ="oakridge_higheast_soilp.xlsx"
#
# # # Reading an excel file
# excelFile = pd.read_excel (inputExcelFile)
#
# # Converting excel file into CSV file
# excelFile.to_csv ("oakridge_colowest.csv", index = None, header=True)

# data = pd.read_csv("oakridge_low_east_soilp.csv")
# data = pd.read_csv("oakridge_orloweast_soilp.csv")
data = pd.read_csv("dcsip_HNNewSheet1.csv")
# data= data.sort_values(by="time").reset_index(drop=True)
from nam.wrapper import NAMRegressor, MultiTaskNAMRegressor
# data = pd.read_csv("new_NO3.csv")
# data= data[(data["Site Name"] == "Agron Farm") | (data["Site Name"] == "Allstar#1")]

# data['datetime']=pd.to_datetime(data[['year', 'month', 'day']])
# df = read_excel("Data_for_ML_crop_residues_assessment.xlsx", sheet_name = 'FinalAnalysis_forManuscript')
# print(df.head())
# for col in data.columns:
#     plt.hist(data[col], bins=30, color='skyblue', edgecolor='black')
#
# # Adding labels and title
#     plt.xlabel('Values')
#     plt.ylabel('Frequency')
#     plt.title(data[col])
#
# # Display the plot
#     plt.show()


#
from sklearn.preprocessing import LabelEncoder
#
# labelencoder = LabelEncoder()
# data["Crop"] = labelencoder.fit_transform(data["Crop"])
# data["Treatment"] = labelencoder.fit_transform(data["Treatment"])
# data["Site Name"] = labelencoder.fit_transform(data["Site Name"])
#
#
# df_crop= pd.get_dummies(data["Crop"])
# df_treatment=pd.get_dummies(data["Treatment"])
#
#  # Merge & drop
# df = pd.concat([data, df_crop], axis=1)
# df= pd.concat([data,df_treatment],axis=1)
# date=data["Date"]
# df['year'] = pd.DatetimeIndex(df['Date']).year
# print(df['year'])
# df['month'] = pd.DatetimeIndex(df['Date']).month
# print(df['month'])

# print(df.columns)
# data = pd.concat([data, df_crop], axis=1).drop(['Crop'], axis=1)
# data=(data.drop(['Crop'],axis=1)).join(df_crop)
# X=X1[['Sand(%)', 'K_factor', 'Crop Rotations_CSW', 'Crop Rotations_CS', 'Co
# X1=pd.get_dummies(data, columns=["Crop","Treatment","Site Name","areaname"])
# X=X1[['Sand(%)', 'K_factor', 'Crop Rotations_CSW', 'Crop Rotations_CS', 'Corn Yield(bu/Ha)',
#       'Soybean Yield(bu/Ha)', 'Slope(%)', 'SlopeLength(m)',
#       'T_factor', 'K_factor', 'Sand(%)', 'Silt(%)', 'Clay(%)', 'Organic matter(%)', 'Rainfall_Erosivity','SCI']]
# X=data[['year','month','day','stemp','atemp','ppt']]
# X= data[['year','month','day','stemp','atemp','prec']].diff().dropna()
# y = data['dsmnrl'].diff().dropna()
# print(df.columns)
# print("Old Shape", data.shape)
# X= data[['day','month','year','stemp','atemp','ppt']]
# data["som2c(1)_lag2"] = data["som2c(1)"].shift(10)
# data["som2c(1)_diff1"] = data["som2c(1)"] - data["som2c(1)"].shift(1)
from statsmodels.tsa.seasonal import seasonal_decompose,STL
# data.set_index("time", inplace=True)

# Decompose the target
# decomposition = seasonal_decompose(data["som2c(1)"], model='additive', period=1)  # adjust period if needed
#
stl = STL(data['fbrchc'], period=12)
decomposition = stl.fit()
# #
stl1 = STL(data['som2c(1)'], period=12)
decomposition1 = stl1.fit()
#
stl2 = STL(data['som2c(2)'], period=12)
decomposition2 = stl2.fit()

stl3 = STL(data['som3c'], period=12)
decomposition3 = stl3.fit()


#
#
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# plot_pacf(data['som3c'])
# plt.show()

# plot_acf(data['som2c(1)'], lags=len(data)//2)
# plt.show()

data["fbrchc_trend"] = decomposition.trend
data["fbrchc_seasonal"] = decomposition.seasonal
data["fbrchc_resid"] = decomposition.resid
# #
data["som2c(1)_trend"] = decomposition1.trend
data["som2c(1)_seasonal"] = decomposition1.seasonal
data["som2c(1)_resid"] = decomposition1.resid

data["som2c(2)_trend"] = decomposition2.trend
data["som2c(2)_seasonal"] = decomposition2.seasonal
data["som2c(2)_resid"] = decomposition2.resid
#
data["som3c_trend"] = decomposition3.trend
data["som3c_seasonal"] = decomposition3.seasonal
data["som3c_resid"] = decomposition3.resid
# # #
data["som2c(1)_trend_lag1"] = data["som2c(1)_trend"].shift(1)
data["som2c(1)_seasonal_lag1"] = data["som2c(1)_seasonal"].shift(1)
data["som2c(1)_resid_lag1"] = data["som2c(1)_resid"].shift(1)
# #
data["som2c(2)_trend_lag1"] = data["som2c(2)_trend"].shift(1)
data["som2c(2)_seasonal_lag1"] = data["som2c(2)_seasonal"].shift(1)
data["som2c(2)_resid_lag1"] = data["som2c(2)_resid"].shift(1)
# #
data["som3c_trend_lag1"] = data["som3c_trend"].shift(1)
data["som3c_seasonal_lag1"] = data["som3c_seasonal"].shift(1)
data["som3c_resid_lag1"] = data["som3c_resid"].shift(1)
#
data["fbrchc_trend_lag1"] = data["fbrchc_trend"].shift(1)
data["fbrchc_seasonal_lag1"] = data["fbrchc_seasonal"].shift(1)
data["fbrchc_resid_lag1"] = data["fbrchc_resid"].shift(1)
#
#
#
data = data.dropna().reset_index(drop=True)
# print(data)
# data=data.fillna(0)
# X= data[['nstemp','natemp','nppt','fbrchc_trend_lag1', "fbrchc_seasonal_lag1","fbrchc_resid_lag1", "som2c(1)_resid_lag1","som2c(1)_seasonal_lag1","som2c(1)_trend_lag1","som2c(2)_resid_lag1","som2c(2)_seasonal_lag1","som2c(2)_trend_lag1","som3c_resid_lag1","som3c_seasonal_lag1","som3c_trend_lag1"]]
time=data['time']
X= data[['time','nstemp','natemp','nppt','fbrchc_trend_lag1', "fbrchc_seasonal_lag1","fbrchc_resid_lag1"]]
# X= data[['time','nstemp','natemp','nppt']]

# X = [col for col in data.columns if col not in ['time', 'som2c(1)']]
# X= data[['fcacc','rlvacc','frtjacc','frtmacc','fbracc','rlwacc','crtacc','rleavc','frootcj','frootcm','fbrchc','rlwodc','crootc','strucc(1)','strucc(2)','metabc(1)','metabc(2)','som1c(1)','som1c(2)','som2c(2)','som3c','dslit','dsom2c(1)','dsom2c(2)','dsom3c','dsomtc','nppt','nstemp','natemp']]

# y = data['NEE']

# data.set_index('time', inplace=True)

y = data['fbrchc']
# y = pd.concat([data['som2c(1)'],data['som2c(2)'], data['som3c']],axis=1)
# y = pd.concat([data['NPP'],data['NEE']],axis=1)


# X=data[['Plot',"SRadiation","Prec",'WD',"WS","AT",'RH',"WC","ST","Crop","Treatment","Site Name","Latitude","Longitude"]]

# Q1 = data['NO3'].quantile(0.25)
# Q3 = data['NO3'].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR
# # #
# # # Create arrays of Boolean values indicating the outlier rows
# upper_array = np.where(data['NO3'] >= upper)[0]
# lower_array = np.where(data['NO3'] <= lower)[0]
# #
# #
# data.drop(index=upper_array, inplace=True)
# data.drop(index=lower_array, inplace=True)
# print("New Shape", data.shape)
# X=df[['Plot',"SRadiation","Prec",'WD',"WS","AT",'RH',"WC","ST","Site Name","Crop","Treatment"]]
#
# y=df['NO3']
# f, ax = plt.subplots(figsize=(10, 10))
# plt.rc('font', size=20)
# # corr_matrix = data[['nstemp', 'natemp', 'nppt','fcacc','fbrchc','som2c(1)','som2c(2)','som3c','dsomtc']].corr().abs()
# corr_matrix = data[["tmax",'tmin','stemp', 'atemp', 'ppt','NPP','NEE']].corr().abs()
# # # Drop features
# sns.heatmap(corr_matrix, annot=True, fmt='.2f',ax=ax)
# plt.show()
from scipy.signal import detrend
# X=detrend(X,type='constant')
# y = detrend(y, type='constant')


from matplotlib import pyplot

# window_size = 5
#
# # Calculate the moving average
# moving_average = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
#
# # Detrend the time series by subtracting the moving average
# yd = y[window_size - 1:] - moving_average
# print(X.shape)
# print(min(data['fcacc']))
# print(max(data['fcacc']))
# print(np.mean(data['fcacc']))
# print(np.var(data['fcacc']))
# print(skew(data['fcacc'], axis=0, bias=True))
# print(kurtosis(data['fcacc'], axis=0, bias=True))

# print(X.describe())


# y.drop(index=upper_array, inplace=True)
# y.drop(index=lower_array, inplace=True)


from sklearn.metrics import mean_absolute_percentage_error
class Stats:

    def __init__(self, X, y, model):
        self.data = X
        self.target = y
        self.model = model
        ## degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        ## degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.model.predict(self.data)) ** 2
        return np.sum(squared_errors)

    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target)
        squared_errors = (self.target - avg_y) ** 2
        return np.sum(squared_errors)

    def r_squared(self):
        '''returns calculated value of r^2'''
        return 1 - self.sse() / self.sst()

    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        return 1 - (self.sse() / self._dfe) / (self.sst() / self._dft)

def pretty_print_stats(stats_obj):
    '''returns report of statistics for a given model object'''
    items = ( ('sse:', stats_obj.sse()), ('sst:', stats_obj.sst()),
             ('r^2:', stats_obj.r_squared()), ('adj_r^2:', stats_obj.adj_r_squared()) )
    for item in items:
        print('{0:8} {1:.4f}'.format(item[0], item[1]))


# X1 = data[['Code', 'areasymbol']].values
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0])
# ohe.fit_transform(X1).toarray()
# data["Code"] = data["Code"].astype('category')
# data["Code"] = data["Code"].cat.codes
# print(data.head())

#
# # Sample the train data set while holding out 20% for testing (evaluating) the classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import shap
from sklearn.metrics import accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# print(X_train.shape)
# print(y_train.shape)
# train_set = df.iloc[:500]
# test_set = df.iloc[500:750]

# X_train, X_test = X.iloc[:12784], X.iloc[12785:22282]
# y_train, y_test = y.iloc[:12784], y.iloc[12785:22282]
X_train, X_test = X.iloc[:409], X.iloc[410:719]
time_train,time_test=time.iloc[:409],time.iloc[410:719]
y_train, y_test = y.iloc[:409], y.iloc[410:719]
# X_train, X_test = X.iloc[:421], X.iloc[422:733]
# y_train, y_test = y.iloc[:421], y.iloc[422:733]
# y_train, y_test = y.iloc[:409], y.iloc[410:719]

# print(X_train)
# print(X_test)
# print(y_test)
# print(X_test)
# print(y_test)
# exit()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
# print(y_test.shape)


# corr_matrix = data[['Plot #','Treatment',"SRadiation","Prec",'WD',"WS","AT","WC","ST",'RH','NO3']].corr().abs()
#
# # Drop features
# sns.heatmap(corr_matrix, annot=True, fmt='.2f')
# plt.show()

from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler

scaler = StandardScaler()
# # # # scaler=MinMaxScaler()
# # # # scaler=RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# pca = PCA(n_components=4)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# y_train = y_train.values.reshape(len(y_train), 1)
# y_test = y_test.values.reshape(len(y_test), 1)
# scaler =MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model=LinearRegression()

# model= Sequential()
# model.add(Dense(50, activation='relu', input_dim=16))
# # model1.add(Dense(25, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))
# model.summary()

# model = Sequential()
# model.add(Dense(128, activation='relu',kernel_initializer='normal', input_dim=16))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256,kernel_initializer='normal', activation='relu'))
# model.add(Dense(1,kernel_initializer='normal',activation='linear'))
# model=linear_model.Ridge()
# from classifier import CascadeForestRegressor
# model = CascadeForestRegressor(random_state=1)
# model = XGBRegressor()
from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
from hyperopt import tpe
from sklearn.metrics import mean_absolute_error
from tabpfn import TabPFNRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
# model = HyperoptEstimator(regressor=any_regressor('xgb'), preprocessing=any_preprocessing('pre'),
#                           loss_fn=mean_absolute_error, algo=tpe.suggest, max_evals=50, trial_timeout=3)
# model = Ridge(alpha=0.5)
# model= linear_model.BayesianRidge()
model = GradientBoostingRegressor()
# model=RandomForestRegressor()
# model=MLPRegressor()
# model = GridSearchCV(svm.SVR(),param_grid,refit=True,verbose=2)
# model = CascadeForestRegressor(random_state=1)
# model = RandomForestRegressor()
# model=ExtraTreesRegressor(max_depth=10)
# model=XGBRegressor()
# model=TabPFNRegressor()
# model=MLPRegressor(hidden_layer_sizes=(100,20), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,
#                  solver='adam', verbose=10, tol=1e-8, learning_rate_init=.01, activation='relu')

# model=DecisionTreeRegressor()
# model=KNeighborsRegressor()
# model = NAMRegressor(
#             num_epochs=1000,
#             num_learners=50,
#             monitor_loss=True,
#             n_jobs=10,
#             # random_state=random_state # random_state is not defined
#         )
# model = MultiOutputRegressor(RandomForestRegressor(random_state=123)).fit(X, y)
# model = MultiOutputRegressor(TabPFNRegressor()).fit(X, y)
# model = MultiOutputRegressor(Ridge(alpha=1.0)).fit(X, y)
# model = MultiTaskNAMRegressor(
#             num_learners=20,
#             patience=60,
#             num_epochs=1000,
#             num_subnets=10,
#             metric='auroc',
#             monitor_loss=False,
#             early_stop_mode='max',
#             n_jobs=10,
#         )
# forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]
#
# model = GridSearchCV(model, forest_params, cv = 10, scoring='accuracy')

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict

# k_folds = KFold(n_splits = 5)
# from sklearn.model_selection import ShuffleSplit
# scores = cross_val_score (model, X, y, cv = k_folds,scoring='neg_mean_absolute_error')
# cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# scores = cross_val_score (model, X, y, cv = cv)
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())

# model=MLPRegressor()


# import lightgbm as ltb
# model = ltb.LGBMRegressor(random_state=42,max_depth=5)

#
# model=SVR()

# model = MLPRegressor(hidden_layer_sizes=(100,10,20), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,
#                     learning_rate_init=.001,activation='relu')

# model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mean_absolute_error'])
# history2 = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
model.fit(X_train,y_train)
# slope = model.coef_[0]  # For simple linear regression, there's one slope
# intercept = model.intercept_
# print(slope)
# print(intercept)
# #
# explainer = shap.Explainer(model.predict, X_test)
# shap_values = explainer(X_test)
# # # #
# # # # # shap.summary_plot(shap_values)
# # # # # or
# # shap.plots.beeswarm(shap_values[:,:,1])
# shap.plots.beeswarm(shap_values)
# plt.rc('font', size=20)
# plt.show()

# Calculates the SHAP values - It takes some time

# mae = model.score(X_test, y_test)

# summarize the best model
# print(model.best_model())
# model.summary()
# joblib.dump(model, "rf_model.pkl")
# predictions1= model.predict(X_train)
# model = joblib.load('rf_model.pkl')
# joblib.dump(model, "rf_model.pkl")
# predictions1= model.predict(X_train)
# model = joblib.load('rf_model.pkl')
# import xlsxwriter as xlsw

predictions1= model.predict(X_train)
predictions= model.predict(X_test)
# y_pred = cross_val_predict(model, X, y, cv=cv)
from pandas import DataFrame
from matplotlib import pyplot

# residuals = predictions - y_test
#
#
# fig, ax = plt.subplots()
# mean=np.array(predictions).mean(axis=0)
# g=ax.scatter(predictions, residuals, edgecolors=(0, 0, 1))
# fig,ax=plt.subplots(figsize=(6,6))
# mean=np.array(predictions).mean(axis=0)
# plt.plot(X,mean,label='Mean at each bootstrap',color='blue')
# plt.xlabel('Y-values',fontsize=15)
# plt.ylabel('X-values',fontsize=15)


# ax.axhline(y=0, color='black', linestyle='--', label='Zero Residual')
# # plt.fill_between(X.to_numpy().flatten(),mean-(2*np.std(np.array(predictions),axis=0)),mean+(2*np.std(np.array(predictions),axis=0)),alpha=0.2)
# plt.legend(fontsize='small')
# # plt.scatter(X,y)
# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
# PredictionErrorDisplay.from_predictions(
#     y,
#     y_pred=y_pred,
#     kind="actual_vs_predicted",
#     subsample=100,
#     ax=axs[0],
#     random_state=0,
# )
# axs[0].set_title("Actual vs. Predicted values")
# PredictionErrorDisplay.from_predictions(
#     y,
#     y_pred=y_pred,
#     kind="residual_vs_predicted",
#     subsample=100,
#     ax=axs[1],
#     random_state=0,
# )
# axs[1].set_title("Residuals vs. Predicted Values")
# fig.suptitle("Plotting cross-validated predictions")
# plt.tight_layout()
# plt.show()

import seaborn as sns
# sns.distplot(residuals, bins = 10) # you may select the no. of bins
# plt.title('Error Terms', fontsize=20)
# plt.xlabel('Residuals', fontsize = 15)
# plt.show()
# mae = -1 * scores.mean()
# deviation = scores.std()
#
# # hard-coded to be 95% confidence interval
# scale = 1.96
# margin_error = mae + scale * deviation
# lower = y_test - margin_error
# upper = predictions + margin_error
#
# print("Lower",lower.mean())
# print("Upper",upper.mean())
#
# plt.figure(figsize=(15, 7))
#
# x = range(predictions.size)
#
# plt.plot(x, predictions, "r--", lw=1, label='predictionTest')
# plt.plot(x, y_test, "b--", lw=1, label='actualTest')
# fill_alpha = 0.5
# fill_color = 'yellow'
# plt.fill_between(x, lower, upper, color=fill_color, alpha=fill_alpha, label='95% CI')
#
# # Plotting anomalies if any
# anomalies = np.array([np.nan] * len(y_test))
# anomalies[y_test < lower] = y_test[y_test < lower]
# anomalies[y_test > upper] = y_test[y_test > upper]
# plt.plot(anomalies, 'o', markersize=5, label='Anomalies')
#
# plt.legend(loc='best')
# plt.tight_layout()
# plt.grid(True)
# plt.show()


# df1=pd.DataFrame(predictions1)


# xlsfile = 'pandas_simple.xlsx'
# writer = pd.ExcelWriter(xlsfile, engine='xlsxwriter')
# df1.to_excel(writer, sheet_name="Sheet1Name",startrow=1, startcol=1, header=False, index=False)
import pandas as pd
# CSV1 = pd.DataFrame({
#     "Prediction": y_train
# })
# CSV = pd.DataFrame({
#     "Prediction": y_test
# })
# CSV2 = pd.DataFrame({
#     "Prediction": predictions1
# })
# CSV3 = pd.DataFrame({
#     "Prediction": predictions
# })

# X_test_new=scaler.inverse_transform(X_test)
# # y_test_new=scaler.inverse_transform(y_test[:,0])
# y_test_new1=scaler.inverse_transform(y_test[:,1].reshape(-1,1))
# print(X_test_new.type)
# inversed = scaler.inverse_transform(X_test)
# CSV4 = pd.DataFrame({
#     # "day": X_test['day'],
#     # "month":X_test['month'],
#     # "year":X_test['year'],
#     "time":inversed[:,0],
#     # "dsmnrl": X_test["dsrfclit"],
#     # "Observed": y_test_new,
#     "Observed1": y_test,
#     "Predicted":predictions
# })

# CSV1.to_csv("y_train.csv", index=False)
# CSV.to_csv("y_test.csv", index=False)
# CSV2.to_csv("prediction_train.csv", index=False)
# # CSV3.to_csv("prediction_test.csv", index=False)
# CSV4.to_csv("prediction_test.csv", index=False)
# X_train.to_csv('./out.csv')
# writer = pd.ExcelWriter('newoutlier_loweast_dsrfclit.xlsx', engine='xlsxwriter')
# CSV4.to_excel(writer, sheet_name='Sheet1')  # Default position, cell A1.
# y_train.to_excel(writer, sheet_name='Sheet1', startcol=7)
# CSV2.to_excel(writer, sheet_name='Sheet1', startcol=8)
# df3.to_excel(writer, sheet_name='Sheet1', startrow=6)

# writer.save()
# print(predictions1)
# print(predictions.shape)
# print(X_test.size)
# print(y_test.size)

# model.predict_prob()
from sklearn.metrics import mean_squared_error


## residuals
# residuals = y_test - predictions
# max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
# max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
# max_true, max_pred = y_test[max_idx], predictions[max_idx]
# print("Max Error:", "{:,.0f}".format(max_error))

fig, ax = plt.subplots()
mean=np.array(predictions).mean(axis=0)
g=ax.scatter(y_test, predictions, edgecolors=(0, 0, 1))

# g.axes.set_yscale('log')
# g.axes.set_xscale('log')
p1 = max(max(predictions), max(y_test))
p2 = min(min(predictions), min(y_test))
plt.plot([p1, p2], [p1, p2], 'r--',lw=2)
plt.xlabel('Observed', fontsize=15)
plt.ylabel('Predicted', fontsize=15)
plt.axis('equal')
plt.show()


# fig, ax = plt.subplots()
# mean=np.array(predictions).mean(axis=0)
# g=ax.scatter(y_test[:,1], predictions[:,1], edgecolors=(0, 0, 1))
# #
# g.axes.set_yscale('log')
# g.axes.set_xscale('log')
# p1 = max(max(predictions), max(y_test))
# p2 = min(min(predictions), min(y_test))
# plt.plot([p1, p2], [p1, p2], 'r--',lw=2)
# plt.xlabel('Observed', fontsize=15)
# plt.ylabel('Predictions', fontsize=15)
# plt.axis('equal')
# plt.show()
# plt.scatter(X['stemp'], y)
# plt.show()

# from sklearn.inspection import PartialDependenceDisplay
#
# fig, ax = plt.subplots(figsize=(15, 10))
# plt.rc('font', size=20)
# # ax.set_title("Partial Dependence Plots")
# PartialDependenceDisplay.from_estimator(
#     estimator=model,
#     X=X_test,
#     features=( 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14),  # the features to plot
#     # categorical_features=(2, 3),  # categorical features
#     random_state=5,
#     target=1,
#     ax=ax,
# )
# plt.show()

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# make sure your X_test has column names (pandas DataFrame preferred)
# new_names = ['Soil_temperature','Average_temperature','Precipitation','fbrchc_trend_lag1', "fbrchc_seasonal_lag1","fbrchc_resid_lag1", "som2c(1)_resid_lag1","som2c(1)_seasonal_lag1","som2c(1)_trend_lag1","som2c(2)_resid_lag1","som2c(2)_seasonal_lag1","som2c(2)_trend_lag1","som3c_resid_lag1","som3c_seasonal_lag1","som3c_trend_lag1"]  # etc.
# X_test_df = pd.DataFrame(X_test, columns=new_names)
# fig, ax = plt.subplots(figsize=(20, 15))
# plt.rc('font', size=16)
#
# PartialDependenceDisplay.from_estimator(
#     estimator=model,
#     X=X_test_df,
#     features=new_names[:15],  # first 15 features
#     target=1,
#     random_state=5,
#     n_cols=5,   # 5 features per row
#     ax=ax
# )
#
# plt.tight_layout()
# plt.show()

# ax.plot([y_test.min(), y_test.max()], [predictions.min(), predictions.max()], 'r--', lw=2)
# plt.fill_between(predictions,mean-(2*np.std(np.array(predictions),axis=0)),mean+(2*np.std(np.array(predictions),axis=0)),alpha=0.2)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# plt.show()


# fig,ax=plt.subplots(figsize=(6,6))
# mean=np.array(predictions).mean(axis=0)
# plt.plot(X,mean,label='Mean at each bootstrap',color='blue')
# plt.xlabel('Y-values',fontsize=15)
# plt.ylabel('X-values',fontsize=15)
# plt.title('95% Confidence Interval of Predictions',fontsize=17,color='red')
# plt.fill_between(X.to_numpy().flatten(),mean-(2*np.std(np.array(predictions),axis=0)),mean+(2*np.std(np.array(predictions),axis=0)),alpha=0.2)
# plt.legend(fontsize='small')
# plt.scatter(X,y)
# plt.tight_layout()
# plt.show()

import numpy as np


# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(20).plot(kind='barh')
# plt.show()

# X_grid = np.arange(min(X_value), max(X_value), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_test, y_test, color = 'red')
# plt.scatter(X_test, predictions, color = 'green')
# plt.title('Random Forest Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Revenue')
# plt.show()

# y_test = y_test.to_numpy().reshape(-1)
# predictions = predictions.reshape(-1)

# mae = metrics.mean_absolute_error(y_test[:,2], predictions[:,2])
mae = metrics.mean_absolute_error(y_test, predictions)
# mape= metrics.mean_absolute_percentage_error(y_test[:,3],predictions[:,3])
# mape= metrics.mean_absolute_percentage_error(y_test,predictions)
# mae1 = metrics.mean_absolute_error(y_train, predictions1)
# print("RMSE",sqrt(mean_squared_error(y_test[:,2], predictions[:,2])))
print("RMSE",sqrt(mean_squared_error(y_test, predictions)))
# print("RMSE Train",sqrt(mean_squared_error(y_train, predictions1)))
# mse = metrics.mean_squared_error(y_test[:,3], predictions[:,3])
# mse = metrics.mean_squared_error(y_test, predictions)
# mse1 = metrics.mean_squared_error(y_train, predictions1)
# r21 = metrics.r2_score(y_test[:,2], predictions[:,2])
# r2 = metrics.r2_score(y_train[:,3], predictions1[:,3])
# r22 = metrics.r2_score(y_test[:,1], predictions[:,1])
r2= metrics.r2_score(y_test, predictions)
# r2_train = metrics.r2_score(y_train, predictions1)
print("MAE",mae)
# print("MAE Train",mae1)
# print("R2 Train",r2_train)
# print("R2",r21)
# print("R2",r22)
print("R2",r2)
#
# #
# print("MSE",mse)

# inversed = scaler.inverse_transform(X_test)

# plt.figure(figsize=(10, 4))
# plt.rc('font', size=20)
# plt.plot(time_test, y_test, label='Observed')       # assuming first column of X_test is "time"
# plt.plot(time_test, predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('som2c(2)')  # looks like you're predicting som2c(1), not som2c(2)
# plt.legend()
# plt.tight_layout()
# plt.show()
# #
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, predictions, edgecolors=(0, 0, 1))
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 45-degree line
# plt.xlabel("Observed")
# plt.ylabel("Predicted")
# # plt.title(f"Actual vs Predicted (R² = {r2:.3f})")
# plt.tight_layout()
# plt.show()


# predictions = pd.Series(predictions, inversed[:,0])

# Combine into a DataFrame
# results = pd.DataFrame({
#     "Observed": y_test,
#     "Predicted": predictions
# })

# Group by year and take the mean
# print(inversed[:,0])
# results_yearly = results.groupby(inversed[:,0].astype(int)).mean()
# #
# # # Plot bar chart
# ax = results_yearly.plot(kind="bar", figsize=(10,6), width=0.7)
# plt.rc('font', size=20)
# plt.xlabel("Year")
# plt.ylabel("Average dsomtc")
# # plt.title("Year-wise Average: Actual vs Predicted")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("MAPE",mape)
# print("MSE train",mse1)

# stats = Stats(X_test, y_test, model)
# pretty_