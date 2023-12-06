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


