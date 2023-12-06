import pandas as pd
import matplotlib.pyplot as plt
import csv

def clean_conditionals(df):
    sep_cols = ["MSZoning","LotShape","LandContour","Utilities","LotConfig","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","Foundation","Heating","Electrical","GarageType","PavedDrive","Fence","MiscFeature","SaleType","SaleCondition"]

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
    df_encoded.to_csv('encoded_df.csv', index=False)

    return df_encoded

# Read in data initially
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

train_df_encoded = clean_conditionals(train_df)
test_df_encoded = clean_conditionals(test_df)

# The following will clean the rest of the data
# Street Column
d_street = {'Grvl': 1, 'Pave':2}
X_train['Street'] = X_train['Street'].apply(d_street.get)

# Alley Column
d_alley = {np.nan: 0, 'Grvl': 1, 'Pave': 2}
X_train['Alley'] = X_train['Alley'].apply(d_alley.get)

# LandSlope Column
d_lslope = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
X_train['LandSlope'] = X_train['LandSlope'].apply(d_lslope.get)

# ExterQual Column
d_exter = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
X_train['ExterQual'] = X_train['ExterQual'].apply(d_exter.get)

# ExterCond Column
X_train['ExterCond'] = X_train['ExterCond'].apply(d_exter.get)

# BsmtQual Column
d_bsmt = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
X_train['BsmtQual'] = X_train['BsmtQual'].apply(d_bsmt.get)

# BsmtCond Column
X_train['BsmtCond'] = X_train['BsmtCond'].apply(d_bsmt.get)

# BsmtExposure
d_bsmtexp = {np.nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
X_train['BsmtExposure'] = X_train['BsmtExposure'].apply(d_bsmtexp.get)

#BsmtFinType1 Column
d_bsmtfin1 = {np.nan: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
X_train['BsmtFinType1'] = X_train['BsmtFinType1'].apply(d_bsmtfin1.get)

# BsmtFinType2 Column
X_train['BsmtFinType2'] = X_train['BsmtFinType2'].apply(d_bsmtfin1.get)

# HeatingQC Column
d_heat = d_exter
X_train['HeatingQC'] = X_train['HeatingQC'].apply(d_heat.get)

#CentralAir Column
d_air = {'Y': 1, 'N': 0}
X_train['CentralAir'] = X_train['CentralAir'].apply(d_air.get)

#KitchenQual Column
d_kitchqual = d_heat
X_train['KitchenQual'] = X_train['KitchenQual'].apply(d_kitchqual.get)

# FireplaceQu Column
d_fire = d_bsmt
X_train['FireplaceQu'] = X_train['FireplaceQu'].apply(d_fire.get)

# GarageQual Column
d_garage = d_fire
X_train['GarageQual'] = X_train['GarageQual'].apply(d_garage.get)

# GarageCond Column
X_train['GarageCond'] = X_train['GarageCond'].apply(d_garage.get)

# PoolQC Column
d_pool = {np.nan: 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
X_train['PoolQC'] = X_train['PoolQC'].apply(d_pool.get)

