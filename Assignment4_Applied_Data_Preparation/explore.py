import pandas as pd
import numpy as np

df = pd.read_csv('Healthcare_Data_Preparation/dataset/diabetic_data.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.replace('?', np.nan, inplace=True)
print("Missing:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nUnique values of some categorical columns:")
for col in ['gender', 'race', 'weight', 'medical_specialty', 'payer_code', 'max_glu_serum', 'A1Cresult']:
    if col in df.columns:
        print(col, df[col].unique())
