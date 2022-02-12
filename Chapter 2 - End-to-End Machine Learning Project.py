# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GETTING THE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# DOWNLOADING THE DATA-------------------------------------------------------------------------------------------------

# IMPORTING THE NEEDED LIBRARIES
import os
import tarfile
import urllib.request

# DEFINING AND ASSIGNING THE ROOT/URL FOR THE DATA REPO ON GITHUB
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# CREATING A FUNCTION TO AUTOMATE THE PROCESS OF FETCHING DATA FROM GITHUB REPOS
# THIS FUNCTION CREATES A DIRECTORY WITH THE DATA WE JUST FETCHED
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# CALLING THE FUNCTION WE CREATED
fetch_housing_data()

# LOADING THE DATA------------------------------------------------------------------------------------------------------

# IMPORTING PANDAS LIBRARY TO HANDLE THE DATAFRAME
import pandas as pd


# DEFINING A FUNCTION TO LOAD THE DATA INTO OUR SCRIPT
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# CALLING THE FUNCTION WE CREATED
housing = load_housing_data()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXPLORING THE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# RETURNING THE TOP FIVE ROWS USING THE HEAD METHOD
print(housing.head())

# RETURNING A DESCRIPTION OF THE DATA USING THE INFO METHOD
# WE NOTICE THAT THE TOTAL_BEDROOMS ATTRIBUTE HAS 20433 NON-NULL VALUES WHILE ALL OTHERS CONTAIN 20640
# WE ALSO NOTICE THAT ALL ATTRIBUTES ARE NUMERICAL EXCEPT FOR THE OCEAN_PROXIMITY WHICH HAS AN OBJECT DATA TYPE
print(housing.info())

# OUR ASSUMPTION IS THAT OCEAN_PROXIMITY ATTRIBUTE IS A CATEGORICAL ATTRIBUTE
# WE CHECK THIS BY CALLING THE VALUE_COUNTS METHOD ON THIS COLUMN
print(housing['ocean_proximity'].value_counts())

# RETURNING A SUMMARY OF THE NUMERICAL ATTRIBUTES USING THE DESCRIBE METHOD
print(housing.describe())

# IMPORTING MATPLOTLIB TO VISUALISE THE DATA
import matplotlib.pyplot as plt

# PLOT A HISTOGRAM FOR THE NUMERICAL ATTRIBUTES
# HISTOGRAMS PLOT THE NUMBER OF INSTANCES AT A GIVEN VALUE RANGE FOR EACH ATTRIBUTE
housing.hist(bins=50, figsize=(20, 15))
plt.show()

""" 
CONCLUSIONS THAT CAN BE MADE FROM THE HISTOGRAM ABOVE:
1 - THE MEDIAN_INCOME ATTRIBUTE IS NOT EXPRESSED IN US DOLLARS. THE UNIT USED IS 1 = $10K
2 - THE MEDIAN_HOUSE_AGE AND MEDIAN_HOUSE_VALUE ARE CAPPED WHICH CAN BE SOLVED BY:
    - COLLECTING PROPER LABELS FOR THE VALUES WHOSE LABELS WERE CAPPED
    - REMOVE THE CAPPED VALUES FROM THE TRAINING AND TESTING SET 
3 - THE ATTRIBUTES HAVE DIFFERENT SCALES
4 - MOST OF THE HISTOGRAMS ARE TAIL HEAVY MEANING THEY EXTEND MUCH FURTHER TO THE RIGHT OF THE MEDIAN THAN TO THE LEFT    
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SPLITTING THE DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# CREATING OUR OWN SPLIT FUNCTION -------------------------------------------------------------------------------------

# IMPORTING NUMPY LIBRARY
import numpy as np

# SETTING A RANDOM SEED VALUE TO MAKE SURE THE SAME EXACT SHUFFLED DATA IS USED EACH TIME WE RERUN THE CODE
np.random.seed(42)

# DEFINING OUR OWN FUNCTION TO SPLIT THE DATA
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# CALLING OUR OWN SPLIT FUNCTION

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))


import hashlib


def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

test_set.head()

# USING PRE-BUILD SCIKIT-LEARN SPLIT FUNCTOIN