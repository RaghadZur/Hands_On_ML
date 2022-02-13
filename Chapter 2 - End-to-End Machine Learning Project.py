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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIAL DATA EXPLORATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
# THE FUNCTION TAKES IN THE DATAFRAME AND TEST RATIO AS INPUT
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# CALLING THE SPLIT FUNCTION WE JUST CREATED
train_set, test_set = split_train_test(housing, 0.2)

# RETURNING THE LENGTHS OF THE THE TRAIN AND TEST SET TO CHECK IF THE RATIO WAS DONE CORRECTLY
print(len(train_set))
print(len(test_set))

# USING PRE-BUILD SCIKIT-LEARN SPLIT FUNCTION--------------------------------------------------------------------------

# IMPORTING THE SPLIT FUNCTION FROM THE SKLEARN LIBRARY
from sklearn.model_selection import train_test_split

# CALLING THE PRE-BUILD FUNCTION
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# STRATIFIED SAMPLING TO OVERCOME BIAS ISSUE IF NEEDED-----------------------------------------------------------------

"""
CONSIDERING A SITUATION WHERE THE MEDIAN INCOME IS AN IMPORTANT ATTRIBUTE TO PREDICT THE HOUSING PRICES, WE WOULD 
NEED TO ENSURE THAT THE TEST SET IS REPRESENTATIVE OF THE VARIOUS CATEGORIES OF INCOME IN THE WHOLE DATASET. 
HOWEVER, SINCE THE MEDIAN INCOME IN A NUMERICAL ATTRIBUTE, WE WILL BE CREATING AN INCOME CATEGORY ATTRIBUTE.
"""

# CREATING A CATEGORICAL ATTRIBUTE FOR THE MEDIAN_INCOME NUMERICAL ATTRIBUTE
housing["income_categ"] = pd.cut(housing["median_income"],
                                 bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                 labels=[1, 2, 3, 4, 5])

# HISTOGRAM OF THE NEW ATTRIBUTE
housing["income_categ"].hist()

# IMPORTING SKLEARN FUNCTION TO PERFORM STRATIFIED SAMPLING
from sklearn.model_selection import StratifiedShuffleSplit

# CALLING THE FUNCTION AND SPLITTING THE DATA
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_categ"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# CHECKING THE PROPORTIONS OF THE CATEGORIES IN THE NEW SPLITTED DATA
print(strat_test_set["income_categ"].value_counts() / len(strat_test_set))

# CHECKING THE PROPORTIONS OF THE CATEGORIES IN THE ORIGINAL DATA TO COMPARE IT TO OUR SPLITTED DATA PROPORTIONS
print(housing["income_categ"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_categ", axis=1, inplace=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VISUALISING AND DISCOVERING THE DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# MAKING A COPY OF THE TRAINING SET (TO PREVENT HARMING THE DATASET)
housing = strat_train_set.copy()

# VISUALISING GEOGRAPHICAL DATA ---------------------------------------------------------------------------------------

"""
THE DATASET CONTAINS TWO GEOGRAPHICAL ATTRIBUTES WHICH ARE LATITUDE AND LONGITUDE WHICH WE CAN USE TO VISUALISE 
THE GEOGRAPHICAL DATA
"""

# PLOTTING A SCATTER PLOT OF THE GEOGRAPHICAL INFO
housing.plot(kind="scatter", x="longitude", y="latitude")

# SETTING ALPHA = 0.1, TO VISUALISE THE PATTERNS BETTER IN THE SCATTER PLOT
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# A MORE DETAILED AND REPRESENTATIVE SCATTER PLOT
# PARAMETER ALPHA IS HOW TRANSPARENT EACH CIRCLE IS
# PARAMETER C STANDS FOR COLOUR, MEANING THE COLOUR OF EACH CIRCLE ON THE PLOT IS DEPENDANT ON THE PRICE VALUE
# PARAMETER S STANDS FOR SIZE, MEANING THE SIZE OF EACH CIRCLE ON THE PLOT IS DEPENDANT ON THE POPULATION IN THE AREA
# PARAMETER CMAP TO SET THE COLOUR SCHEME SINCE WE HAVE PARAMETER C
housing.plot(kind="scatter",
             x="longitude",
             y="latitude",
             alpha=0.3,
             c="median_house_value",
             s=housing["population"] / 100,
             cmap=plt.get_cmap("jet")
             )
plt.show()

# PLOTTING THE GEOGRAPHICAL SCATTER PLOT ON THE MAP IMAGE---------------------------------------------------------------

# DOWNLOADING THE MAP IMAGE INTO OUR DIRECTORY
images_path = os.path.join("C:/Users/ragha/OneDrive/Desktop/Hands_On_ML", "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# IMPORTING THE NEEDED LIBRARY
import matplotlib.image as mpimg

# READING THE IMAGE OF THE MAP INTO THE SCRIPT
california_img = mpimg.imread(os.path.join(images_path, filename))

# SAME AS PREVIOUSLY DONE, CREATED A DETAILED PLOT
ax = housing.plot(kind="scatter",
                  x="longitude",
                  y="latitude",
                  figsize=(10, 7),
                  s=housing['population'] / 100,
                  label="Population",
                  c="median_house_value",
                  cmap=plt.get_cmap("jet"),
                  colorbar=False,
                  alpha=0.4)

# SETTING THE BACKGROUND MAP IMAGE WITH HALF TRANSPARENCY BY SETTING ALPHA TO 0.5 PLUS ADDING AXIS
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))

# SETTING THE X AND Y LABELS FOR THE PLOT
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

# SETTING THE COLOURBAR
prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values / prices.max())
cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)

# LABELLING THE COLOURBAR AND SHOWING THE PLOT
cbar.set_label('Median House Value', fontsize=16)
plt.legend(fontsize=16)
plt.show()

# LOOKING FOR CORRELATIONS --------------------------------------------------------------------------------------------

# CORRELATION MATRIX BETWEEN EVERY ATTRIBUTE
correlation_matrix = housing.corr()

# SINCE WE ARE INTERESTED IN HOUSE VALUES, WE CAN CHECK THE CORRELATION BETWEEN EACH ATTRIBUTE AND THE HOUSE VALUES
print(correlation_matrix["median_house_value"].sort_values(ascending=False))

""" 
We can notice from the above the following:
1 - median_income attribute has a strong positive correlation with the median_house_value
2 - total_bedrooms and population attribute has nearly no correlation with the median_house_value
3 - latitude attribute has a small negative correlation with the median_house_value
"""

# IMPORTING NEEDED LIBRARIES TO VISUALISE CORRELATION MATRIX
from pandas.plotting import scatter_matrix

# SETTING THE ATTRIBUTES WE WANT TO VISUALISE
attributes = ["median_house_value", "median_income", "total_bedrooms", "latitude"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()