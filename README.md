<H3>ENTER YOUR NAME : VAISHNAVI S</H3>
<H3>ENTER YOUR REGISTER NO : 212222230165</H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
Developed by: VAISHNAVI S
RegisterNumber: 212222230165

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
# DATASET:
![307534704-564dbb82-2f4f-4a86-b8d9-681f46ea1519](https://github.com/user-attachments/assets/a18a3060-d638-47cc-bba6-7d5f6ec0ce99)


# DROPPING THE UNWANTED DATASET:
![307534746-9bb25efd-5226-46ea-aef1-de170cbf1f6c](https://github.com/user-attachments/assets/0c994607-2e9d-4085-b2d1-16a4f74c8add)


# CHECKING NULL VALUES:
![307534766-91474fee-123d-4175-b258-92bbf75723cb](https://github.com/user-attachments/assets/8ac12fef-60a0-454c-a4ad-6d8f22f6ab01)

# CHECKING FOR DUPLICATION:
![307534818-90c71c87-f527-4e00-9c0d-3f36b6748e67](https://github.com/user-attachments/assets/9c11ed25-5f3a-4ceb-b04a-68da5ed753b3)

# DESCRIBING THE DATASET:
![307534831-02c414f1-451f-4baa-9298-afd1b4d1fedf](https://github.com/user-attachments/assets/43fed966-2cfb-4d0a-9659-7eee10dc525e)

# SCALING THE DATASET:
![307534852-9da6e306-00ef-469d-94b0-162f0e51acdb](https://github.com/user-attachments/assets/af11264f-a752-4811-8d02-0a51f032cad6)

# X FEATURES:
![307534875-d86e8b16-6046-4b4b-ae39-d4321e70d102](https://github.com/user-attachments/assets/a757cb73-f30d-4a0f-bc41-6c03fc5e0ddd)

# Y FEATURES:
![307535219-5b0431f6-6b7d-4f30-acee-e92da9a0a4a3](https://github.com/user-attachments/assets/3aed7b6f-a5f6-4cb1-94f4-159dd3d87277)

# SPLITTING THE TRAINING AND TESTING DATASET:

![307534908-5f7fb6ae-6e05-4a1b-b4b5-201e51ccb22d](https://github.com/user-attachments/assets/1a04ed27-cd95-4dec-a1a3-e66545941811)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


