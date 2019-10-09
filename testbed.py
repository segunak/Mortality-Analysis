import sys
import warnings
import numpy as np
import pandas as pd
import json
from numpy import genfromtxt
#from tabulate import tabulate
import numpy.linalg as linalg
import matplotlib.pyplot as plt
#from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('C:/GitHub/Mortality-Analysis/mortality-data/2005_data.csv')
print(df['130_infant_cause_recode'].mode()[0]) 

most_frequent_death = str(df['130_infant_cause_recode'].mode()[0]).zfill(3)

with open('C:/GitHub/Mortality-Analysis/mortality-data/2005_codes.json') as f:
       codes = json.load(f)

for code in codes['130_infant_cause_recode']:
    if code == most_frequent_death: 
        print(code[0])

print(codes['130_infant_cause_recode']['004'])      
print("DSSFDSFDSFSFDS")