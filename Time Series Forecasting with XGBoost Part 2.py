import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

#1. Load Data 
df = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\Time Series Forecasting with XGBoost\PJME_hourly.csv")

#2. Change the index 
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

#3. Plot overall graph 
df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='PJME Energy Use in MW')
plt.show()

#3a. Previous Train/Test Split
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

#3a. Plot one week's data  
df.loc[ (df.index > '01-01-2010') & (df.index < '01-07-2010')].sort_index().plot(figsize = (15,5), title = 'One Week Data in 2010')
plt.show()

#4. Outlier Analysis 
#2013 looks like outlier, may be blackout or wrong recording 

df['PJME_MW'].plot(kind = 'hist', bins = 500)
plt.show()


df.query('PJME_MW < 20_000' )
df.query('PJME_MW < 20_000' ).plot(kind = 'hist')
plt.show()

df.query('PJME_MW < 20_000' ).plot(figsize = (15,5), style = '.')
plt.show()

df.query('PJME_MW < 19_000')['PJME_MW'] .plot(style='.',
          figsize=(15, 5),
          color=color_pal[5],
          title='Outliers')
plt.show()

#5. Remove Outlier 
df = df.query('PJME_MW > 19_000').copy()

df

#6. Time Series Cross Validation 
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits= 5, test_size= 24*365*1, gap= 24 )
df = df.sort_index() 

        
train_idx
val_idx


fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

fold = 0

for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['PJME_MW'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['PJME_MW'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.show()


#7. Forecasting Horizon

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



#8. Lag Features - What was the target (x) days in the past.

def add_lags(df):
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

df = add_lags(df)

#9. Train Using Cross Validation 

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3']
    TARGET = 'PJME_MW'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

#Finished 
scores

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


#10. Predicting the Future
df = create_features(df)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'PJME_MW'

X_all = df[FEATURES]
y_all = df[TARGET]

reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)

# Predicting starting at the max date 
df.index.max() #Timestamp('2018-08-03 00:00:00')

# Create future dataframe
future = pd.date_range('2018-08-03','2019-08-01', freq='1h') #One Year Prediction 
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])
df_and_future
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)


future_w_features = df_and_future.query('isFuture').copy()



#Predict the future 
future_w_features['pred'] = reg.predict(future_w_features[FEATURES])

future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1,
                               lw=1,
                               title='Future Predictions')
plt.show()


#Saving Model For Later 
reg

reg.save_model('model.json')


#!ls -lh

reg_new = xgb.XGBRegressor()
reg_new.load_model('model.json')
future_w_features['pred'] = reg_new.predict(future_w_features[FEATURES])
future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1, lw=1,
                               title='Future Predictions')
