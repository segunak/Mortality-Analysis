#!/usr/bin/env python
# coding: utf-8 
# Author: Segun Akinyemi

# ## Data & Library Imports
# This section contains the various libraries and data files that we're used throughout the code. 
import re
import sys
import nltk 
import json
import copy
import warnings
import numpy as np
import pandas as pd
import texttable as tt
from numpy import genfromtxt
from tensorflow import keras
from tabulate import tabulate
import numpy.linalg as linalg
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from prettytable import PrettyTable
from tensorflow.keras import layers
import statsmodels.api as statsmodels
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, wordpunct_tokenize

# Ensuring that output warnings are not displayed and setting some formatting options
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,}'.format

cdc_data_2005 = pd.read_csv('./mortality-data/2005_data.csv', na_values=['NA','?'])
cdc_data_2006 = pd.read_csv( './mortality-data/2006_data.csv', na_values=['NA','?'])
cdc_data_2007 = pd.read_csv( './mortality-data/2007_data.csv', na_values=['NA','?'])
cdc_data_2008 = pd.read_csv( './mortality-data/2008_data.csv', na_values=['NA','?'])
cdc_data_2009 = pd.read_csv( './mortality-data/2009_data.csv', na_values=['NA','?'])
cdc_data_2010 = pd.read_csv( './mortality-data/2010_data.csv', na_values=['NA','?'])
cdc_data_2011 = pd.read_csv( './mortality-data/2011_data.csv', na_values=['NA','?'])
cdc_data_2012 = pd.read_csv( './mortality-data/2012_data.csv', na_values=['NA','?'])
cdc_data_2013 = pd.read_csv( './mortality-data/2013_data.csv', na_values=['NA','?'])
cdc_data_2014 = pd.read_csv( './mortality-data/2014_data.csv', na_values=['NA','?'])
cdc_data_2015 = pd.read_csv( './mortality-data/2015_data.csv', na_values=['NA','?'])

# Importing ICD Codes
with open ('./mortality-data/2005_codes.json') as json_file: 
    icd_codes_2005 = json.load(json_file)   
with open ('./mortality-data/2006_codes.json') as json_file: 
    icd_codes_2006 = json.load(json_file)
with open ('./mortality-data/2007_codes.json') as json_file: 
    icd_codes_2007 = json.load(json_file)
with open ('./mortality-data/2008_codes.json') as json_file: 
    icd_codes_2008 = json.load(json_file)
with open ('./mortality-data/2009_codes.json') as json_file: 
    icd_codes_2009 = json.load(json_file)
with open ('./mortality-data/2010_codes.json') as json_file: 
    icd_codes_2010 = json.load(json_file)
with open ('./mortality-data/2011_codes.json') as json_file: 
    icd_codes_2011 = json.load(json_file)
with open ('./mortality-data/2012_codes.json') as json_file: 
    icd_codes_2012 = json.load(json_file)
with open ('./mortality-data/2013_codes.json') as json_file: 
    icd_codes_2013 = json.load(json_file)
with open ('./mortality-data/2014_codes.json') as json_file: 
    icd_codes_2014 = json.load(json_file)
with open ('./mortality-data/2015_codes.json') as json_file: 
    icd_codes_2015 = json.load(json_file)


# ## Function Definitions
# This section defines some functions that are used repeatedly throughout our analysis.

# **Function Definition: Retrieving the full name of an ICD code.**
def FindFullNameFromCode(target_code, icd_codes): 
    target_code = str(target_code).zfill(3)
    for code in icd_codes: 
        if code == target_code: 
            return icd_codes[code]


# **Function Definition: Finding the single most frequent cause of death in a data set**
def MostFrequentCauseOfDeath(data, icd_codes): 
    most_frequent_death = str(int(data.mode()[0])).zfill(3)
    for code in icd_codes: 
        if code == most_frequent_death: 
            return icd_codes[code]


# **Function Definition: Finding the top `n` causes of death in a data set** 
def TopCausesOfDeath(cdc_data, icd_descriptions, n = 10):
    
    deathsByFrequency = cdc_data.value_counts()
    top_n_deaths = deathsByFrequency.head(n).rename_axis('Code').reset_index(name='Deaths')
    
    codeDescriptions = [icd_descriptions[code] for code in top_n_deaths['Code']]
    top_n_deaths["Description"] = codeDescriptions
    
    return top_n_deaths


# **Function Definition: Creating a dictionary of ICD codes and their associated named descriptions.**
def MapIcdToDesc(cdc_data, icd_codes): 
    codeToDescDict = {}
    
    for code in set(cdc_data): 
        zeroPaddedCode = str(code).zfill(3)
        codeToDescDict.update({
            code: icd_codes[icd] 
            for icd in icd_codes if icd == zeroPaddedCode
        }) 

    return codeToDescDict


# **Function Definition: Checking the results of POS tagging. Prints out pre-tag and post-tag lists for comparison**
def CheckTaggingResults(pre_tagging_list, tagged_list): 
    for index, desc in enumerate(pre_tagging_list):
        if(len(desc) != len(tagged_list[index])): 
            print("Pre-Tagging:", desc)
            print("Post-Tagging:", tagged_list[index], "\n\n")

# Creating dictionaries to hold {key, value} pairs containing {icd code, description} for each year. 
icd_desc_2005 = MapIcdToDesc(cdc_data_2005["113_cause_recode"], icd_codes_2005["113_cause_recode"])
icd_desc_2006 = MapIcdToDesc(cdc_data_2006["113_cause_recode"], icd_codes_2006["113_cause_recode"])
icd_desc_2007 = MapIcdToDesc(cdc_data_2007["113_cause_recode"], icd_codes_2007["113_cause_recode"])
icd_desc_2008 = MapIcdToDesc(cdc_data_2008["113_cause_recode"], icd_codes_2008["113_cause_recode"])
icd_desc_2009 = MapIcdToDesc(cdc_data_2009["113_cause_recode"], icd_codes_2009["113_cause_recode"])
icd_desc_2010 = MapIcdToDesc(cdc_data_2010["113_cause_recode"], icd_codes_2010["113_cause_recode"])
icd_desc_2011 = MapIcdToDesc(cdc_data_2011["113_cause_recode"], icd_codes_2011["113_cause_recode"])
icd_desc_2012 = MapIcdToDesc(cdc_data_2012["113_cause_recode"], icd_codes_2012["113_cause_recode"])
icd_desc_2013 = MapIcdToDesc(cdc_data_2013["113_cause_recode"], icd_codes_2013["113_cause_recode"])
icd_desc_2014 = MapIcdToDesc(cdc_data_2014["113_cause_recode"], icd_codes_2014["113_cause_recode"])
icd_desc_2015 = MapIcdToDesc(cdc_data_2015["113_cause_recode"], icd_codes_2015["113_cause_recode"])

# Finding the top 10 causes of death in the United States for each year. 
top_ten_2005 = TopCausesOfDeath(cdc_data_2005["113_cause_recode"], icd_desc_2005)
top_ten_2006 = TopCausesOfDeath(cdc_data_2006["113_cause_recode"], icd_desc_2006)
top_ten_2007 = TopCausesOfDeath(cdc_data_2007["113_cause_recode"], icd_desc_2007)
top_ten_2008 = TopCausesOfDeath(cdc_data_2008["113_cause_recode"], icd_desc_2008)
top_ten_2009 = TopCausesOfDeath(cdc_data_2009["113_cause_recode"], icd_desc_2009)
top_ten_2010 = TopCausesOfDeath(cdc_data_2010["113_cause_recode"], icd_desc_2010)
top_ten_2011 = TopCausesOfDeath(cdc_data_2011["113_cause_recode"], icd_desc_2011)
top_ten_2012 = TopCausesOfDeath(cdc_data_2012["113_cause_recode"], icd_desc_2012)
top_ten_2013 = TopCausesOfDeath(cdc_data_2013["113_cause_recode"], icd_desc_2013)
top_ten_2014 = TopCausesOfDeath(cdc_data_2014["113_cause_recode"], icd_desc_2014)
top_ten_2015 = TopCausesOfDeath(cdc_data_2015["113_cause_recode"], icd_desc_2015)

# Finding top deaths of all years combined and sorting by death count, highest to lowest. 
topTenAllYears = pd.concat([top_ten_2005, top_ten_2006, top_ten_2007, top_ten_2008, top_ten_2009, top_ten_2010,
                           top_ten_2011, top_ten_2012, top_ten_2013, top_ten_2014, top_ten_2015])\
                   .groupby(['Code']).sum()\
                   .sort_values(by='Deaths', ascending=False)\
                   .reset_index()

# Re-adding Description column after calculation and formatting death data with commas. 
topTenAllYears['Description'] = [icd_desc_2015[code] for code in topTenAllYears['Code']]
#topTenAllYears['Deaths'] = topTenAllYears['Deaths'].apply("{:,}".format)

# The 125s is just adding spaces to center the header. 
print('{:^125s}'.format("Top 12 Deaths in the United States from 2005 - 2015"))
topTenAllYears.style.hide_index()

ax = topTenAllYears.plot.barh(x="Description", y="Deaths")

# Plot of All other diseases against age. 
x111 = cdc_data_2015["age_recode_27"].values
plt.title("All other diseases")
plt.xlim(0, 26)
plt.ylim(0,600000)
binblock = np.arange(1, 28, 1)
plt.hist(x111, bins = binblock, rwidth=0.9)
plt.xlabel("age_recode_27")
plt.ylabel("Numbers")
plt.show()

# Plot of Ischemic heart disease against age. 
x63 = cdc_data_2015["age_recode_27"].values
plt.title("All other forms of chronic ischemic heart disease")
plt.xlim(0, 26)
plt.ylim(0,500000)
binblock=np.arange(1, 28, 1)
plt.hist(x63, bins = binblock, rwidth=0.9)
plt.xlabel("age_recode_27")
plt.ylabel("Numbers")
plt.show()

# Plot of Malignant neoplasms of trachea, bronchus and lung aganist age. 
x27 = cdc_data_2015["age_recode_27"].values
plt.title("Malignant neoplasms of trachea, bronchus and lung")
plt.xlim(0, 26)
plt.ylim(0,300000)
binblock=np.arange(1, 28, 1)
plt.hist(x27 ,bins = binblock, rwidth=0.9)
plt.xlabel("age_recode_27")
plt.ylabel("Numbers")
plt.show()

# Plot of Cerebrovascular diseases aganist age. 
x70 = cdc_data_2015["age_recode_27"].values
plt.title("Cerebrovascular diseases")
plt.xlim(0, 26)
plt.ylim(0,350000)
binblock=np.arange(1, 28, 1)
plt.hist(x70 ,bins = binblock, rwidth=0.9)
plt.xlabel("age_recode_27")
plt.ylabel("Numbers")
plt.show()

# Plot of Acute myocardial infarction againist age. 
x59 =cdc_data_2015["age_recode_27"].values
plt.title("Acute myocardial infarction")
plt.xlim(0, 26)
plt.ylim(0,250000)
binblock=np.arange(1, 28, 1)
plt.hist(x59 ,bins = binblock, rwidth=0.9)
plt.xlabel("age_recode_27")
plt.ylabel("Numbers")
plt.show()

# Importing Medical Transcript Data and a pre-made binary file for vectorizing our data. 
medical_data = pd.read_csv('./data-from-canvas/medicaltranscriptions.csv', sep=',', header=0)
pub_med_model = KeyedVectors.load_word2vec_format('./data-from-canvas/PubMed-and-PMC-w2v.bin', binary=True)
stop_words = set(stopwords.words('english'))

# Ignore this. Used to check that we aren't losing data
def check(): 
    print("Tagged Desriptions", len(medical_desc_tagged))
    print("Tokenixed Desriptions", len(medical_desc_tokenized))
    print("Long Desc", len(long_descriptions))
    print("766 Long Desc", long_descriptions[766])
    print("766 Tokenized", medical_desc_tokenized[766])        
    print("766 Tagged", medical_desc_tagged[766])

# Creating a list of lists, where each internal list is a tokenized description, one for each patient description in the 
# medicaltranscripts.csv file. 
patientId = 0
long_descriptions = {}
medical_desc_tokenized = []

for desc in medical_data['description']:
    if(desc.isspace()): 
        continue
    long_descriptions[patientId] = desc
    token_desc = word_tokenize(desc) 
    token_desc = [word for word in token_desc if word]
    token_desc = [word.lower() for word in token_desc] 
    token_desc = [word.lower() for word in token_desc if not word in stop_words] 
    token_desc = [word.lower() for word in token_desc if word.isalpha()]
    medical_desc_tokenized.append(token_desc)
    patientId += 1

# Removing duplicates from each individual description, for example "heart pain heart" becomes "heart pain". 
medical_desc_tokenized = [list(set(desc_list)) for desc_list in medical_desc_tokenized]

# Part of Speech tagging for each word in each patient description. We keep nouns, verbs and adjectives. We tried
# this with only nouns and found that adding verbs and adjectives gives us better cosine similarity scores. 
accpetable_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBG']
medical_desc_tagged = copy.deepcopy(medical_desc_tokenized)

for index, desc_list in enumerate(medical_desc_tagged): 
    token_list = nltk.pos_tag(desc_list)
    for word, pos in token_list:
        if (pos not in accpetable_tags):
            desc_list.remove(word)
    if(len(desc_list) == 0): 
        del medical_desc_tagged[index]
        del long_descriptions[index]

# All possible POS tags returned by this function. 
# nltk.help.upenn_tagset()
#pd.set_option('display.max_colwidth', -1)

# Using a function to check the results of POS tagging. Prints each description before & after tagging.
CheckTaggingResults(medical_desc_tokenized, medical_desc_tagged)

# Tokenizing and cleaning up the ICD code descriptions. We don't want the uneccesary stuff in there. . 
icd_desc_tokenized = []
for desc in topTenAllYears['Description']: 
    token_desc = word_tokenize(desc) 
    token_desc = [word for word in token_desc if word]
    token_desc = [word for word in token_desc if word.isalpha()]
    icd_desc_tokenized.append(token_desc)

icd_desc_tokenized

# Creating a new list of lists, where each internal list is a patient visit description with ONLY words that can be 
# converted to a vector. 
medical_desc_modeled = []
for desc in medical_desc_tagged: 
    desc_list = [word for word in desc if word in pub_med_model]
    medical_desc_modeled.append(desc_list)

# Calculating a similarity score between the ICD code description and the description of each patients medical visit. We
# use a cutoff similarity score of 0.40, or 40%. Meaning, the cosine similarity between the patients visit description and
# an ICD code description must be 40% or higher for us to consider it in our final output. This is our similarity measure
# metric 

patientId = 0
similarities_and_desc = []

for desc in medical_desc_modeled: 
    patient_desc_sentence = ' '.join(word for word in desc)
    for code_desc in icd_desc_tokenized:
        cosine_sim = pub_med_model.n_similarity(desc, code_desc)
        if(cosine_sim < 0.4): 
            continue
        icd_desc_sentence = ' '.join(word for word in code_desc)
        data = (patientId, patient_desc_sentence, icd_desc_sentence, cosine_sim)
        similarities_and_desc.append(data)
    patientId += 1


# Creating a final output data frame, sorting by highest similarity scores to lowest, and converting from a pos tagged
# list description back to the long form sentence descriptions, for display purposes. 

final_patient_data = pd.DataFrame(similarities_and_desc, columns = ["Patient Id", 
                                                                    "Visit Description", 
                                                                    "ICD Code Description", 
                                                                    "Cosine Similarity"])

# Converting from the tokenized description back to the long form description.
for index, row in final_patient_data.iterrows(): 
    final_patient_data.loc[index, "Visit Description"] = long_descriptions[row["Patient Id"]]

final_patient_data = final_patient_data.sort_values("Cosine Similarity", ascending=False)
final_patient_data["Cosine Similarity"] = ["{0:.0%}".format(sim) for sim in final_patient_data["Cosine Similarity"]]
pd.set_option('display.max_rows', 45000)
pd.set_option('display.max_colwidth', -1)

# Length of 3864 with 0.40 as the cutoff similarity score. 
final_patient_data.style.hide_index()

