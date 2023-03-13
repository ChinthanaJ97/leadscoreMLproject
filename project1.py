#Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# visulaisation
from matplotlib.pyplot import xticks
get_ipython().run_line_magic('matplotlib', 'inline')

# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# Data Preparation

#     1. Data loading


#Importing .csv file
data = pd.read_csv("L:/MLseries/dataset/dataset/assignment/DataScienceInternship.csv")
data.head() 

data.shape

data.info()

data.describe()

# 2. Data preprocess


#checking duplicates in whole rows
duplicate_rows = sum(data.duplicated())
duplicate_rows



#checking duplicates in "Agent_id"
sum(data.duplicated(subset = 'Agent_id')) == 0


# duplicates exist in "Agent_id"


#checking duplicates in "Agent_id" and "lead_id"
sum(data.duplicated(subset = ['Agent_id',"lead_id"]))==0



# duplicates exist in "Agent_id" and "lead_id"





data.duplicated(subset = ['Agent_id',"lead_id"])




# removing duplicates exists in "Agent_id" and "lead_id"
data=data.drop_duplicates(subset=['Agent_id','lead_id'])

# size of dataframe after removing duplicates
data.shape





sum(data.duplicated(subset = ['Agent_id',"lead_id"]))==0





# Converting '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0' values to NaN.
data = data.replace('9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0', np.NaN)
data



#finding number of missing values in each column
data.isna().sum()




# percentage of missing values in each column 
round(100*(data.isnull().sum()/len(data.index)), 2)



#here dropping "Unnamed: 0" column
data.drop(columns=["Unnamed: 0"], inplace=True)
data


#identify status other than LOST and WON
status_types= data["status"].value_counts()
print(status_types)



#dropping all rows with status other than LOST and WON
data.drop(data[data['status'] == 'OPPORTUNITY'].index, inplace = True)
data.drop(data[data['status'] == 'CONTACTED'].index, inplace = True)
data.drop(data[data['status'] == 'PROCESSING'].index, inplace = True)
data.drop(data[data['status'] == 'IMPORTANT'].index, inplace = True)



# size after removing all rows with status other than LOST and WON
data.shape




round(100*(data.isnull().sum()/len(data.index)), 2)



status_types= data["status"].value_counts()
print(status_types)


# # Handelling missing values


# 1. room_type     

data["room_type"].value_counts()

# visualizing "room_type" in dataframe

fig, ax = plt.subplots(figsize=(20, 10))
 

sns.countplot(data['room_type'],ax=ax)
plt.show()



# replacing missing values of "room_type" with "others", a new type since missing value is 50.42% assigning to mode will change the data behaviour therefor assigned to a new value called others

data['room_type'] = data['room_type'].replace(np.NaN, 'Others')
print(data["room_type"])



data["room_type"].value_counts()


# visualizing "room_type" after correction

fig, ax = plt.subplots(figsize=(20, 10))
 

sns.countplot(data['room_type'],ax=ax)
plt.show()


# 2. Handling source_city

data["source_city"].describe()


data["source_city"].isna().sum()



# replacing source_city missing values with mode

data['source_city'] = data['source_city'].replace(np.NaN, 'ecc0e7dc084f141b29479058967d0bc07dee25d9690a98ee4e6fdad5168274d7')


data["source_city"].describe()


# 3. Handling source_country


data["source_country"].describe()



data["source_country"].isna().sum()



# replacing source_country missing values with mode

data['source_country'] = data['source_country'].replace(np.NaN, 'e09e10e67812e9d236ad900e5d46b4308fc62f5d69446a9750aa698e797e9c96')



data["source_country"].describe()


# 4. Handling des_city



data["des_city"].describe()


data["des_city"].isna().sum()



# replacing des_city missing values with mode

data['des_city'] = data['des_city'].replace(np.NaN, 'ecc0e7dc084f141b29479058967d0bc07dee25d9690a98ee4e6fdad5168274d7')



data["des_city"].describe()


# 5. Handling des_country


data["des_country"].describe()


data["des_country"].isna().sum()



# replacing des_country missing values with mode

data['des_country'] = data['des_country'].replace(np.NaN, '8d23a6e37e0a6431a8f1b43a91026dcff51170a89a6512ff098eaa56a4d5fb19')

data["des_country"].isna().sum()



# cheking mising value percentage
round(100*(data.isnull().sum()/len(data.index)), 2)


# 6. Handling source 


data["source"].describe()


data["source"].isna().sum()

# replacing "source" missing values with mode

data["source"] = data["source"].replace(np.NaN, '7aae3e886e89fc1187a5c47d6cea1c22998ee610ade1f2b7c51be879f0c37ca8')


data["source"].isna().sum()


# Handling lost_reason        6.70

data["lost_reason"].describe()

data["lost_reason"].isna().sum()

# replacing "lost_reason" missing values with mode

data["lost_reason"] = data["lost_reason"].replace(np.NaN, 'Low availability')


data["lost_reason"].isna().sum()

# Handling budget 7.87

data["budget"].describe()


data["budget"].isna().sum()


# replacing "budget" missing values with mode range

data["budget"] = data["budget"].replace(np.NaN, '£60 - £120 Per week')


data["budget"].isna().sum()


# Handling lease              5.01

data["lease"].describe()


# replacing "lease" missing values with mode

data["lease"] = data["lease"].replace(np.NaN, 'Full Year Course Stay 40 - 44 weeks')


data["lease"].isna().sum()


# Handling utm_medium         6.87

data["utm_medium"].describe()

# replacing "utm_medium" missing values with mode

data["utm_medium"] = data["utm_medium"].replace(np.NaN, '09076eb7665d1fb9389c7c4517fee0b00e43092eb34821b09b5730c41ebcc50c')


data["utm_medium"].isna().sum()


round(100*(data.isnull().sum()/len(data.index)), 2)


# Handling movein


data["movein"].describe()


# visualizing movein
fig, ax = plt.subplots(figsize=(50, 15))
 

sns.countplot(data['movein'],ax=ax)
plt.show()


# from excel , 
# newest date - 04/09/24
# oldest date - 01/01/70


data["movein"].value_counts()

# selecting range 10/09/22  to 31/08/22 (the most used data)

# generating random date for given date range

import pandas as pd
import random
from datetime import date, timedelta

def random_date(start, end, seed=1):
    dates = pd.date_range(start, end).to_series()
    return random.choice(dates)

output = random_date("20220831", "20220910", seed=1)
print(output.strftime("%d/%m/%Y"))


# generating random dates for NaN values in data['movein'] , 
# replacing NaN values as string "null" to detection
data['movein'] = data['movein'].replace(np.NaN, 'null')

new=[]
for i in data['movein']:
    
    if i == "null" :
       output = random_date("20220831", "20220910", seed=1)
       x=output.strftime("%d/%m/%Y")
       new.append(x)
    else:  
       new.append(i)
       
print(new)  


# replacing existing movein data with generated values in the new()

data["movein"] = new


data["movein"].isna().sum()

# visualizing movein after handling missing values
fig, ax = plt.subplots(figsize=(50, 15))
 

sns.countplot(data['movein'],ax=ax)
plt.show()



round(100*(data.isnull().sum()/len(data.index)), 2)


# for utm_source 0.13% missing values - As a rule of thumb, if less than 5% of the observations are missing, 
# the missing data can simply be deleted without any significant ramifications 

data.dropna(inplace = True)
round(100*(data.isnull().sum()/len(data.index)), 2)


# final data set size
data.shape

#  Exploratory Analysis - for visualizing status variable("status" indicates whether a lead approches WON or LOST )


# visualizing "status" 
fig, ax = plt.subplots(figsize=(50, 15))
 

sns.countplot(data['status'],ax=ax)
plt.show()


# Creating dummy variables for categorial variable set

# Creating a dummy variable for categorical variables(independent variables) except "agent_id", "lead_id"(ids are unique vaiables) and "status"(status is the dependent variable) 

dummy= pd.get_dummies(data[['lost_reason','budget','lease','movein','source','source_city','source_country','utm_source','utm_medium','des_city','des_country','room_type']])
dummy.head()

# combining cleaned dataframe with dummy varable table
data = pd.concat([data, dummy], axis=1)
data.head()


# removing independent categorical variables from combined dataframe
data = data.drop(['lost_reason','budget','lease','movein','source','source_city','source_country','utm_source','utm_medium','des_city','des_country','room_type'], axis = 1)
data.head()

# feature selection--defining X dataframe to train purpose(including only dummy variable set)

X = data.drop(['Agent_id','lead_id','status'], axis=1)
X.head()


# feature selection--defining Y dataframe for train purpose
Y=data["status"]
Y.head()

# Converting string values of "status" to numerical value

a=[]
for i in data["status"]:
    if i=="LOST":
        i=0
        a.append(i)
    else:
        i=1
        a.append(i)
        
data["status"]= a
#print(a)
print(data["status"])



# assigning numerical "status" to Y
Y= data['status']



# # Training ( on 30 percent of data)  and Testing ( on 70 percent of data)


# Splitting X, Y data into train and test(defines X_train, X_test, Y_train, Y_test dataframes)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)


# Training the data using, random forest model(inputs-  X_train and Y_train dataframes)

from sklearn.ensemble import RandomForestClassifier


num_estimators = 100
min_samples = 4

rf = RandomForestClassifier(n_estimators=num_estimators,min_samples_split=min_samples)
rf.fit(X_train, Y_train.ravel())

# using rf()) function trained on X_train and Y_train data to the prediction 

# input X_test data to rf() to get prediction data set, Y_test_predicted
Y_test_predicted = rf.predict(X_test)
    


# # 4.Evaluating performance using metrics (accuracy, precision, recall and F1-score)

from sklearn import metrics

# accuracy check for Y predicted data vs actual Y test data
accuracy = metrics.accuracy_score(Y_test, Y_test_predicted)

# ROC AUC score check for Y predicted data vs actual Y test data
# (auc score ranges between 0 and 1, aucscore of 0.5-random guess, 1-perfect classifier)
auc_score = metrics.roc_auc_score(Y_test, Y_test_predicted)

# output accuracy and auc_score
print(accuracy)
print(auc_score)



from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score

# accuracy check
print('Accuracy:', accuracy_score(Y_test, Y_test_predicted))

# precision check
print('Precision:', precision_score(Y_test, Y_test_predicted))

# recall check(0-1)
# 1-all positive samples correctly predicted as positive, 0-no positive samples predicted as positive.
print('Recall:', recall_score(Y_test, Y_test_predicted))

# F1 score check(0-1)
# 1-perfect precision and recall,0-either precision or recall is 0.
print('F1 score:', f1_score(Y_test, Y_test_predicted))


# classification report and confussion matrix output
print('\n clasification report:\n', classification_report(Y_test,Y_test_predicted))
print('\n confussion matrix:\n',confusion_matrix(Y_test,Y_test_predicted))