import pandas as pd
import matplotlib.pyplot as plt
import csv

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# Reads a column
column_with_data = {}

train_columns = train_df.columns
for i in train_columns:
    column_with_data[i] = train_df[i]

print(column_with_data)

# Reads each row with all of its data
all_rows = []
my_dict = {}
id_with_row = {}

with open('test.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        for i in row:
            row_split = i.split(",")
            all_rows.append(row_split)

for i in all_rows[1:]:
    id_with_row[i[0]] = i[1:]








# print(test_df.head())
# print(train_df.head())
df_encoded = pd.get_dummies(train_df, columns=['HouseStyle'], prefix='HouseStyle', drop_first=True)

