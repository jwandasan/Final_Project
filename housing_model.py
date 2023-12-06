import pandas as pd
import matplotlib.pyplot as plt
import csv

# Reading a csv file into workable format
all_rows = []
my_dict = {}
id_with_row = {}
column_with_row = {}

with open('test.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        for i in row:
            row_split = i.split(",")
            all_rows.append(row_split)

# for i in all_rows[1:]:
#     for j in i[1:]:
#         id_with_row{f"{}"}



# test_df = pd.read_csv("test.csv")
# train_df = pd.read_csv("train.csv")

# print(test_df.head())
# print(train_df.head())
df_encoded = pd.get_dummies(train_df, columns=['HouseStyle'], prefix='HouseStyle', drop_first=True)

