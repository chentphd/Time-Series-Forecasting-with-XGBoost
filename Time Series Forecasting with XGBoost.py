import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns  


#1. Load Data
df = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\Time Series Forecasting with XGBoost\PJME_hourly.csv")

df.head() 

df.tail()

sorted(df)

df.dtypes

#2. Set index 
#df.set_index("Datetime")
df = df.set_index('Datetime')



#3. Change to DateTime 
#df.index
#pd.to_datetime(df.index)
df.index = pd.to_datetime(df.index)


#4. Graph 
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='PJME Energy Use in MW')
plt.show()

#5. Train / Test Split  - Train before Jan 2015
df.index < '01-01-2015'
df.loc[df.index < '01-01-2015' ]

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

#6. Plot Test and Train
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

#7. Plot One Week's Data 
#df.index > '01-01-2010' & df.index < '01-07-2010'
#df.loc[ (df.index > '01-01-2010') & (df.index < '01-07-2010')]
#df.loc[ (df.index > '01-01-2010') & (df.index < '01-07-2010')].sort_index()
df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].sort_index().plot(figsize=(15, 5), title='1st Week Of Data in 2010 ')
plt.show()


#8. Feature Creation 
#df.index.hour
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

#9. Visualize our Feature / Target Relationship 
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues')
ax.set_title('MW by Month')
plt.show()



#10.  Creating the Model 
from sklearn.metrics import mean_squared_error
import xgboost as xgb 


train = create_features(train)
test = create_features(test)

df.columns

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


#11. Fit through Model 
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)#Every 100th tree 
#Model is built 

#12. Feature Importance 
reg.feature_importances_
pd.DataFrame( data = reg.feature_importances_,
            index = reg.feature_names_in_,
            columns= ['importance'])
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])

fi.sort_values('importance')
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()


#13. Predicting / Forecasting on Test
#reg.predict(X_test)
test['prediction'] = reg.predict(X_test)

#14. Merge with original data frame 
df.merge(test[['prediction']], how = 'left', left_index = True, right_index = True )
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
df

#15. Plot our prediction vs real life 
ax = df[['PJME_MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

#16. Look at only ONE WEEK of Prediction vs Real Data 
ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'].sort_index() \
    .plot(figsize=(15, 5), title='Week Of Data in 2018')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'].sort_index() \
    .plot(style='-')
plt.legend(['Truth Data','Prediction'])
plt.show()

#17. Root MSE 
score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}') # Lower RMSE = Better

#18. Calculate Error 
test[TARGET] - test['prediction']
np.abs( test[TARGET] - test['prediction'] )

test['error'] = np.abs( test[TARGET] - test['prediction'] )

test.index.date
test['date'] = test.index.date


test.groupby('date')['error'].mean()
#Worst days
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)

#Best days
test.groupby(['date'])['error'].mean().sort_values(ascending=True).head(10)





