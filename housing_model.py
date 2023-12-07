import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error 

plt.style.use("ggplot")

## Cleans the conditionals - giving them their own column and making them 0 or 1
def clean_conditionals(df):
    sep_cols = ["MSZoning","Functional","LotShape","LandContour","Utilities","LotConfig","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","Foundation","Heating","Electrical","GarageType","PavedDrive","Fence","MiscFeature","SaleType","SaleCondition"]

    df_encoded = pd.DataFrame()

    # Give conditionals their own column of True or False
    for i in sep_cols:
        encoded = pd.get_dummies(df, columns=[i], prefix = i)
        df_encoded = pd.concat([df_encoded, encoded], axis=1)
        
    # Remove duplicates
    df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
    # Change all Trues and Falses to 1 and 0 respectively
    df_encoded = df_encoded.map(lambda x: 1 if x == True else (0 if x == False else x))
    # Output to CSV file to check
    df_encoded.drop(sep_cols, axis = 1, inplace=True)
    # df_encoded.to_csv('encoded_df.csv', index=False)

    return df_encoded

## Cleans the rest of the data - uncondiotional points
def clean_rest(df):
    # The following will clean the rest of the data
    # Street Column
    d_street = {'Grvl': 1, 'Pave':2}
    df['Street'] = df['Street'].apply(d_street.get)

    # Alley Column
    d_alley = {np.nan: 0, 'Grvl': 1, 'Pave': 2}
    df['Alley'] = df['Alley'].apply(d_alley.get)

    # LandSlope Column
    d_lslope = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
    df['LandSlope'] = df['LandSlope'].apply(d_lslope.get)

    # ExterQual Column
    d_exter = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df['ExterQual'] = df['ExterQual'].apply(d_exter.get)

    # ExterCond Column
    df['ExterCond'] = df['ExterCond'].apply(d_exter.get)

    # BsmtQual Column
    d_bsmt = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df['BsmtQual'] = df['BsmtQual'].apply(d_bsmt.get)

    # BsmtCond Column
    df['BsmtCond'] = df['BsmtCond'].apply(d_bsmt.get)

    # BsmtExposure
    d_bsmtexp = {np.nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df['BsmtExposure'] = df['BsmtExposure'].apply(d_bsmtexp.get)

    #BsmtFinType1 Column
    d_bsmtfin1 = {np.nan: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    df['BsmtFinType1'] = df['BsmtFinType1'].apply(d_bsmtfin1.get)

    # BsmtFinType2 Column
    df['BsmtFinType2'] = df['BsmtFinType2'].apply(d_bsmtfin1.get)

    # HeatingQC Column
    d_heat = d_exter
    df['HeatingQC'] = df['HeatingQC'].apply(d_heat.get)

    #CentralAir Column
    d_air = {'Y': 1, 'N': 0}
    df['CentralAir'] = df['CentralAir'].apply(d_air.get)

    #KitchenQual Column
    d_kitchqual = d_heat
    df['KitchenQual'] = df['KitchenQual'].apply(d_kitchqual.get)

    # FireplaceQu Column
    d_fire = d_bsmt
    df['FireplaceQu'] = df['FireplaceQu'].apply(d_fire.get)

    # GarageQual Column
    d_garage = d_fire
    df['GarageQual'] = df['GarageQual'].apply(d_garage.get)

    # GarageCond Column
    df['GarageCond'] = df['GarageCond'].apply(d_garage.get)

    # GarageFinish Column
    d_gf = {np.nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    df['GarageFinish'] = df['GarageFinish'].apply(d_gf.get)

    # PoolQC Column
    d_pool = {np.nan: 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df['PoolQC'] = df['PoolQC'].apply(d_pool.get)

## User Input for program
print("*Must have same features as kaggle*")
test_input = input("Enter the name of the test csv file, in the form of 'file.csv':")
train_input = input("Enter the name of the train csv file, in the form of 'file.csv':")

## Read in data initially
test_df = pd.read_csv(test_input)
train_df = pd.read_csv(train_input)

## Clean the data separating categorical values into unique binary columns
train_df_encoded = clean_conditionals(train_df)
test_df_encoded = clean_conditionals(test_df)

##Cleaning categorical columns that have relation to one another into set integer range
clean_rest(train_df_encoded)
clean_rest(test_df_encoded)

## Clears true NaN values
train_df_encoded = train_df_encoded.drop(columns=['LotFrontage', 'GarageYrBlt']) # these are removed since there exists a lot of null values.
test_df_encoded = test_df_encoded.drop(columns=['LotFrontage', 'GarageYrBlt']) # these are removed since there exists a lot of null values.
train_df_encoded = train_df_encoded.dropna()
test_df_encoded = test_df_encoded.dropna()

## Creating our features and targets
X = train_df_encoded.iloc[:, train_df_encoded.columns != 'SalePrice']
Y = train_df_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.shape, y_train.shape)

## Normalizes our integer into values from 0 to 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

## Create the model
def determine_lasso_coefficients(alphas, X, y):
    l_coefs = []
    for a in alphas:
        model = LassoCV(n_alphas=a).fit(X,y)
        l_coefs.append(model.coef_)

    return l_coefs

# Determine alphas
alphas = np.arange(10,101, 10)
lassoCoeffs = determine_lasso_coefficients(alphas, X_train, y_train)   

## Used to create lasso weighting graph, and to validate which n_alpha was better (20 or 60)
# ax = plt.gca()
# ax.plot(alphas, lassoCoeffs)
# plt.ylim((-1000,1000))
# plt.ylabel('WEIGHTS')
# plt.xlabel("ALPHAS FOR LASSOCV()")
# plt.title("Lasso Coefficients as a function of Regularization")
# ax.axvline(x = 20, ymin = -1000, ymax = 1000, ls= '--', color = 'k')
# ax.axvline(x = 60, ymin = -1000, ymax = 1000, ls= '--', color = 'k')
# plt.show()

# # features_seen = [LassoCV(n_alphas=20).fit(X_train, y_train).n_features_in_, LassoCV(n_alphas=60).fit(X_train, y_train).n_features_in_]
# alpha = [LassoCV(n_alphas=20).fit(X_train, y_train).alpha_, LassoCV(n_alphas=60).fit(X_train, y_train).alpha_]
scores = []
for i in ([20,60]):
    lassoEst = LassoCV(n_alphas= i).fit(X_train, y_train)
    print(F"ALPHA = {i}")

    for feature, coef in zip(X.columns, lassoEst.coef_):
        print(F"The magnitude of the feature coefficient foe {feature} is {abs(coef)}")
    
    score = cross_val_score(lassoEst, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_score = -score.mean()
    scores.append(avg_score)

print(scores)
lassoEst = LassoCV(n_alphas=20).fit(X_train, y_train)
print(f"The mean squared error when training the data with LassoCV embedding is {np.sqrt(mean_squared_error(y_train, lassoEst.predict(X_train)))}")
print(f"The mean squared error when testing the data with LassoCV embedding is {np.sqrt(mean_squared_error(y_test, lassoEst.predict(X_test)))}")

print(f"The r^2 score when training the data with LassoCV embedding is {r2_score(y_train, lassoEst.predict(X_train))}")
print(f"The r^2 score when testing the data with LassoCV embedding is {r2_score(y_test, lassoEst.predict(X_test))}")

## Sorting values, ID number is not a interpretable description of the data
y_train_sorted = y_train.sort_values(ascending=True)
y_test_sorted = y_test.sort_values(ascending=True)

# print(f"training: {y_train_sorted.shape}, {lassoEst.predict(X_train).shape}")
# print(f"testing: {y_test_sorted.shape}, {lassoEst.predict(X_test).shape}")

## Unsorted relationship between LassoCV embedding and the train/test values
plt.scatter(np.arange(0,1161), lassoEst.predict(X_train)/ (10**3), label = 'predicted training values')
plt.scatter(np.arange(0,1161), y_train / (10**3), label = 'true training values')
plt.xlabel("House iD [dimensionless]")
plt.ylabel("Housing Price [k$]")
plt.title("LassoCV Predicted Data vs. Model training Data")
plt.legend()
plt.show()
plt.scatter(np.arange(0,291), lassoEst.predict(X_test)/ (10**3), label = 'predicted testing values')
plt.scatter(np.arange(0,291), y_test/ (10**3), label = 'true testing values')
plt.title("LassoCV Predicted Data vs. Model Testing Data")
plt.xlabel("House iD [dimensionless]")
plt.ylabel("Housing Price [k$]")
plt.legend()
plt.show()

## Sorted relationship between LassoCV embedding and the train/test values
plt.scatter(np.arange(0,1161), np.sort(lassoEst.predict(X_train))/ (10**3), label = 'predicted training values')
plt.scatter(np.arange(0,1161), y_train_sorted / (10**3), label = 'true training values')
plt.xlabel("Sorted Houses [dimensionless]")
plt.ylabel("Housing Price [k$]")
plt.title("LassoCV Predicted Data vs. Model training Data")
plt.legend()
plt.show()
plt.scatter(np.arange(0,291), np.sort(lassoEst.predict(X_test))/ (10**3), label = 'predicted testing values')
plt.scatter(np.arange(0,291), y_test_sorted/ (10**3), label = 'true testing values')
plt.title("LassoCV Predicted Data vs. Model Testing Data")
plt.xlabel("Sorted Houses [dimensionless]")
plt.ylabel("Housing Price [k$]")
plt.legend()
plt.show()
