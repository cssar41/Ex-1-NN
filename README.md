<H3>ENTER YOUR NAME: Saravanan C</H3>
<H3>ENTER YOUR REGISTER NO.: 212222110041</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
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
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
## The Dataset:

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/c2ba1d96-f8ba-46d8-9b99-67c454af4fba)

## Dropping unwanted features

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/f03ed636-bdc0-4315-a407-5490d75b06c9)

## Checking for null values

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/af07a56e-d1e8-465d-a861-6a5fd6483a93)

## Checking for duplication

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/62e4ee4c-7baa-4553-82d1-a70b46e7ef62)

## Describing the dataset

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/75cda078-b795-4ac9-92cd-7579048da859)

## Scaling the values

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/97eb666f-0deb-40cc-8a74-dc8e02017915)


## X Features

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/91be32c5-5bf5-4c44-a2c4-08845020828b)

## Y Features

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/214b5709-3d92-45a4-8472-d35af126d98e)

## Splitting the training and testing dataset

![image](https://github.com/Prasanth9025/Ex-1-NN/assets/118343686/adb96d96-e270-46a1-b85d-71e7841ce99e)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


