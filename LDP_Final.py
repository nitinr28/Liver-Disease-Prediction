#!/usr/bin/env python
# coding: utf-8

# In[85]:


# # # Liver Disease Prediction


# In[86]:


# Import all required libraries for reading data, analysing and visualizing data to perform data pre-proccessing


# In[87]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# In[88]:


# # # Data Pre-Proccessing


# In[89]:


# # Data Analysis


# In[90]:


# Read the training & test data by loading the dataset


# In[91]:


liver_df = pd.read_csv("patients.csv")


# In[92]:


liver_df


# In[93]:


liver_df.head()


# In[94]:


liver_df.tail()


# In[95]:


# given in the website 416 liver disease patients and 167 non liver disease patients
# 2- non liver patient and 1 is liver patient 


# In[96]:


liver_df.describe(include='all')


# In[97]:


# Checking for null values in the dataset


# In[98]:


liver_df.info()


# In[99]:


# Here is the observation from the dataset:   
# 1) Only gender is non-numeric veriable. All others are numeric.   
# 2) There are 10 features and 1 output - dataset.   
# 3) In Albumin and Globulin ration we can see four missing values


# In[100]:


liver_df['Dataset'].value_counts()


# In[101]:


# Here we get a clear idea of 416 patients who have liver disease represented as 1 and 167 patients who do not have are represented by 2


# In[102]:


# Let us first fill in the null values of the dataset rather than dropping the whole row


# In[103]:


liver_df.columns


# In[104]:


liver_df.isnull().sum()


# In[105]:


# Clearly shows 4 null values 


# In[106]:


# We will now take the mean of Albumin and Globulin ratio and fill them in the null values 


# In[107]:


liver_df['Albumin_and_Globulin_Ratio'].mean()


# In[108]:


liver_df['Albumin_and_Globulin_Ratio'] = liver_df['Albumin_and_Globulin_Ratio'].fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())


# In[109]:


# Thus the null values have been filled and can be checked


# In[110]:


liver_df.isnull().sum()


# In[111]:


# By the information we can see that there are no null values 


# In[112]:


# # Data Cleaning 


# In[113]:


# Let us check for duplicate values and rows in the dataset 


# In[114]:


duplicateRowsDF = liver_df[liver_df.duplicated(keep='first')]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# In[115]:


duplicateRowsDF = liver_df[liver_df.duplicated(keep='last')]
print("Duplicate Rows except last occurrence based on all columns are :")
print(duplicateRowsDF)


# In[116]:


liver_df.shape # These are the total rows and columns in the dataset


# In[117]:


liver_df = liver_df.drop_duplicates()
print( liver_df.shape )


# In[118]:


# 13 duplicate values are deleted 


# In[119]:


# Let us now check for outliers in the datasets and try correcting them 


# In[120]:


# # Removing Outliers and seeing for each column


# In[121]:


liver_df.columns


# In[122]:


# # Distribution of Numerical Features


# In[123]:


# Plot histogram grid
liver_df.hist(figsize=(15,15), xrot=-45, bins=10) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()


# In[124]:


liver_df.describe()


# In[126]:


# It seems there is outlier in Aspartate_Aminotransferase as the max value is very high than mean value


# In[127]:


# Thus let us try eliminating outliers


# In[128]:


sns.boxplot(liver_df.Total_Bilirubin)


# In[129]:


liver_df.Total_Bilirubin.sort_values(ascending=False).head()


# In[130]:


# These outliers values will not matter since one of the highest recorded bilirubin count is 80 mg/dl
# Since 75 is the highest and can be seen in a liver disease patient we will be keeping it


# In[131]:


sns.boxplot(liver_df.Direct_Bilirubin)


# In[132]:


liver_df.Direct_Bilirubin.sort_values(ascending=False).head()


# In[133]:


# Any direct bilirubin is a part of Total Bilirubin and constitutes of only 10% of the total bilirubin this is mostly negligable


# In[134]:


sns.boxplot(liver_df.Alkaline_Phosphotase)


# In[135]:


# The ALP levels ranged from 1,005 to 3,067 IU/L so even this would not matter 


# In[136]:


sns.boxplot(liver_df.Alamine_Aminotransferase)


# In[137]:


# The measures of ALT can be  >1,000 IU/l thus the outliers can be considered and the difference between the outliers is neglegeble


# In[138]:


sns.boxplot(liver_df.Aspartate_Aminotransferase)


# In[139]:


liver_df.Aspartate_Aminotransferase.sort_values(ascending=False).head() # Here the difference of values is very high so we can eleminate some of the outliers


# In[140]:


liver_df = liver_df[liver_df.Aspartate_Aminotransferase <=3000 ] # We have dropped out one row 
liver_df.shape


# In[141]:


sns.boxplot(liver_df.Aspartate_Aminotransferase)


# In[142]:


liver_df.Aspartate_Aminotransferase.sort_values(ascending=False).head() # We can observe another outliers and this can also be elimated


# In[143]:


liver_df = liver_df[liver_df.Aspartate_Aminotransferase <=2500 ]
liver_df.shape


# In[144]:


# Hence we have eliminated two rows in outlier elimination of Aspartate_Aminotransferase


# In[145]:


liver_df.shape


# In[146]:


sns.boxplot(liver_df.Total_Protiens)


# In[147]:


# These outliers will not create any hinderence to the model


# In[148]:


sns.boxplot(liver_df.Albumin)


# In[149]:


# We can obsereve that there are no outliers present 


# In[150]:


sns.boxplot(liver_df.Albumin_and_Globulin_Ratio)


# In[151]:


# By this observation we can see that these outliers would not cause any hinderence to the model thus can be used


# In[152]:


# Data Visualization


# In[153]:


count_classes = pd.value_counts(liver_df['Dataset'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Liver disease classes histogram")
plt.xlabel("Dataset")
plt.ylabel("Frequency")


# In[154]:


# 2- non liver patient and 1 is liver patient 


# In[155]:


liver_df['Dataset'].value_counts()


# In[156]:


sns.countplot(data=liver_df, x = 'Gender', label='Count')

M, F = liver_df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[157]:


sns.factorplot (x="Age", y="Gender", hue="Dataset", data=liver_df);


# In[158]:


# Age seems to be a factor for liver disease for both male and female genders


# In[159]:


sns.catplot(x="Age", y="Gender", hue="Dataset", data=liver_df);


# In[160]:


liver_df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=True)


# In[161]:


g = sns.FacetGrid(liver_df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[162]:


# So here we can clearly observe that at the age of 40 to 60 people diagnosed with liver disease is very high


# In[163]:


# # # Feature Selection Observation


# In[164]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[165]:


# In this scatter plot we can observe that all the values are in a linear line this shows that there is a direct relationship between both Direct and Total Bilirubin


# In[166]:


# # # Feature Selection Start


# In[167]:


sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=liver_df, kind="reg")


# In[168]:


# Here we can almost observe a linear line which represents high similarity in both Direct and Total Bilirubin


# In[169]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[170]:


# There is linear relationship between Aspartate_Aminotransferase and Alamine_Aminotransferase and the gender. But not as much as Total and Direct Bilirubin


# In[171]:


sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=liver_df, kind="reg")


# In[172]:


# The jointplot for Alamine_Aminotransferase and Aspartate_Aminotranferase shows lower linear relationship than expected


# In[174]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[175]:


# To confirm if there is co-relation between Alamine_Aminotransferase and Alkaline_Phosphatase we such have a joint plot


# In[176]:


sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=liver_df, kind="reg")


# In[177]:


# The joinplot shows no linear correlation between Alkaline_Phosphotase and Alamine_Aminotransferase


# In[178]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[179]:


# There is linear relationship between Total_Protiens and Albumin and the gender. 


# In[180]:


sns.jointplot("Total_Protiens", "Albumin", data=liver_df, kind="reg")


# In[181]:


# There is a linear relationship and can be considered for dropping only under analysis 


# In[182]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[183]:


# There is linear relationship between Albumin_and_Globulin_Ratio and Albumin.


# In[184]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=liver_df, kind="reg")


# In[185]:


# Thus by this information we can observe that the jointplot has lesser linear relationship than showed in scatter plot 


# In[186]:


g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[187]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Total_Protiens", data=liver_df, kind="reg")


# In[188]:


# By this plot we can observe that the linear relationship is lower than expected.


# In[190]:


# # Observation
#From the above jointplots and scatterplots, we find direct relationship between the following features:
#Direct_Bilirubin & Total_Bilirubin
#Aspartate_Aminotransferase & Alamine_Aminotransferase
#Total_Protiens & Albumin
#Albumin_and_Globulin_Ratio & Albumin
#Total_Protiens & Albumin_and_Globulin_Ratio


# In[191]:


# # Label Encoding


# In[192]:


# Have to remap the class labels for convenience, no liver disease then:=0 for patients having liver disease then:=1


# In[193]:


liver_df['Dataset'] = liver_df['Dataset'].map({2:0,1:1})


# In[194]:


liver_df['Dataset'].value_counts()


# In[195]:


# # Encoding -2  Male and Female


# In[196]:


def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df


# In[197]:


liver_df = binary_encode(liver_df, 'Gender', 'Male')


# In[198]:


liver_df.head(3)


# In[199]:


# 1 is male and 0 is female


# In[201]:


# # Corelation


# In[202]:


liver_df.corr()


# In[204]:


plt.figure(figsize=(30, 30))
sns.heatmap(liver_df.corr(), cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.title('Correlation between features');


# In[205]:


#The above correlation also indicates the following correlation
# Total_Protiens & Albumin
# Alamine_Aminotransferase & Aspartate_Aminotransferase
# Direct_Bilirubin & Total_Bilirubin
# There is some correlation between Albumin_and_Globulin_Ratio and Albumin. But its not as high as Total_Protiens & Albumin


# In[206]:


# Now train test split can be done directly or direct bilirubin can be dropped and then train test split can be done


# In[ ]:




