# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 23:25:12 2025

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv("Project - Tobacco compsumption.csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

print(df.info())
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

X = df[['Sex', 'Age', 'Tobacco_consume', 'Smoking']]
y = df['IMC']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

plt.figure(figsize=(6,4))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color':'red'})
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

sm.qqplot(model.resid, line='45', fit=True)
plt.title("Normal Q-Q Plot")
plt.show()