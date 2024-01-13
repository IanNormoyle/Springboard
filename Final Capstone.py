#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First, import files and packages, both the dataset and pandas and numpy
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn.preprocessing as preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score,log_loss
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

df = pd.read_csv(r"C:\Users\Ian's Second PC\Downloads\train.csv")
df.head()


# In[2]:


# In[2]:
#Capstone EDA
#These are additional statistics and basic analysis, looking into features of the dataset post-wrangling
def first_EDA(df):
    size = df.shape
    sum_duplicates = df.duplicated().sum()
    sum_null = df.isnull().sum().sum()
    is_NaN = df. isnull()
    row_has_NaN = is_NaN. any(axis=1)
    rows_with_NaN = df[row_has_NaN]
    count_NaN_rows = rows_with_NaN.shape
    return print("Samples: %d,\nFeatures Count: %d,\nDuplicates: %d,\nNull Entries: %d,\nNumber of Rows with Null Entries: %d %.1f%%" %(size[0],size[1], sum_duplicates, sum_null,count_NaN_rows[0],(count_NaN_rows[0] / df.shape[0])*100))


# In[3]:


first_EDA(df)
#Oddly enough, there are a large amount of null values.


# In[4]:


df2 = df.dropna(axis='columns')


# In[5]:


first_EDA(df2)
#Null values have been removed, for the sake of determining potential best fit variables for the analysis.


# 

# In[6]:


df2pricecheck = df2[['SalePrice', 'YrSold']]


# In[7]:


Frame06 = df2pricecheck[(df2pricecheck['YrSold']==2006)]  
Frame07 = df2pricecheck[(df2pricecheck['YrSold']==2007)]  
Frame08 = df2pricecheck[(df2pricecheck['YrSold']==2008)]  
Frame09 = df2pricecheck[(df2pricecheck['YrSold']==2009)]  
Frame10 = df2pricecheck[(df2pricecheck['YrSold']==2010)]  
print(Frame06['SalePrice'].mean(), Frame07['SalePrice'].mean() ,Frame08['SalePrice'].mean(), Frame09['SalePrice'].mean(), Frame10['SalePrice'].mean())


# In[8]:


df2["RemodelRecency"] = df2["YrSold"]-df2["YearRemodAdd"]
#A new varaible has been added, for the sake of determining whether or not the recency of the remodel itself is impactful
#on the sale price.


# In[9]:


df2["RemodelRecency"].head()


# In[10]:


#As for the typical gap between remodel and sale...
plt.hist(df2['RemodelRecency'])
#The recency tends to be extreme, with the single largest of ten classes done in the five years before the sale, and the
#numbers falling off especially harshly after 15, with a significant rise far in the past, 55-60 years ago. 


# In[11]:


#As far as correlation goes, there are few strongly correlated variables, either in the positive or negative.
#When it comes to the correlated variables, the predictors of sale price are as expected. There is a strong correlation
#which is positive to sale price with high square footages, with lot area and square footage sticking out.
#Similarly, newer houses have a strong tendency to sell for more money, another intuitively true statement. 
#One odd finding is that a higher condition score is negatively correlated with price. It is possible that that score is 
#structured with one as the highest, however the histogram indicates this is unlikely, as there are no ones and several tens.
df2.corr()


# In[12]:


#Notably, Remodel Recency is negatively correlated with SalePrice at 51%, this intuitively follows, as more recently
#remodeled houses have lower scores.


# In[13]:


Essential = df2[['SalePrice', 'OverallQual', 'OverallCond','YearBuilt','YearRemodAdd', 'GarageArea','YrSold','RemodelRecency']]
Essential.head()


# In[14]:


#Next, add the data for the GDP and Inflation information.
econdata = pd.read_csv(r"C:\Users\Ian's Second PC\Documents\GDPANDINFLATIONDATA.csv")


# In[15]:


MinValue = Essential["YrSold"].min()
print(MinValue)


# In[16]:


MaxValue = Essential["YrSold"].max()
print(MaxValue)


# In[17]:


for i in econdata["Year"]: 
   econdata["Year"] = econdata["Year"].replace(i, i-1)
econdata.head()


# In[18]:


Essential2 = pd.merge(Essential, econdata, left_on='YrSold', right_on='Year', how ='left')
Essential2 = Essential2.drop("Year", axis=1)
Essential2.head()


# In[19]:


#Next, the model for the data, respecting year sold, must be determined. The test will begin with the heatmap determining
#correlations for the 'Essential' dataframe: 
Essential2.corr()


# In[20]:


#Of the new variables, the correlation is somewhat low with SalePrice, as GDP Growth represents only 7/10ths of 1%, and 
#inflation is similarly low, at 3% correlated. While the cause of this is not immediately apparent, it may simply be that
#the majority of buyers are purchasing for personal use, and those with the means to purchase a house will not be affected
#by changes in economic conditions to the extent that the situation changes.


# In[21]:


#Visualization: 


# In[22]:


df2pricecheckavg = df2pricecheck.mean()
df2pricecheckavg.plot(x = "YrSold", y ="SalePrice")


# In[23]:


plt.hist(Essential2['SalePrice'])


# In[24]:


#Sale Price is distributed heavily on the low end, with the overwhelming majority falling between 100,000 and 200,000 dollars


# In[25]:


plt.hist(Essential2['OverallQual'])


# In[26]:


#Quality is average, with the peak at 5, but high numbers of houses at 6 and 7.


# In[27]:


plt.hist(Essential2['YearBuilt'])


# In[28]:


#Houses trend overwhelmingly recent, with over half from 1960 or later, and the year 2000 seeing the largest plurality,
#implying a fast-growing market.


# In[29]:


plt.hist(Essential2['YearRemodAdd'])


# In[30]:


#Remodels are most commonly done in the past decade, or between 1950 and 1960. This implies some demand for certain aesthetics,
#but that under most circumstances owners and buyers desire more recent technology.


# In[31]:


plt.hist(Essential2['GarageArea'])


# In[32]:


#Garages tend to be smaller, with the majority under 700 square feet, and only a few above 1000.


# In[33]:


plt.hist(Essential2['YrSold'])


# In[34]:


#Sales drop off entirely in the final six months of 2007, implying that the real estate market's downturn in that period
#affected the local market in Ames as well.


# In[35]:


plt.hist(Essential2['RemodelRecency'])


# In[36]:


#The recency score tends toward extremely recent remodels, implying that remodeling prior to selling is a common
#practice. 


# In[37]:


plt.scatter(Essential2['SalePrice'],Essential2["YrSold"])


# In[38]:


#Cost is tighter in distribution over time, with the very low cost and very high cost sales both disappearing between 2006
#and 2010, most likely due to the economic downturn making it harder to find cheaper housing, and consumer suspicion of 
#more costly options.


# In[39]:


plt.scatter(Essential2['GDP growth (annual %)'], Essential2['SalePrice']) 
plt.xlabel("GDP Growth")  # add X-axis label 
plt.ylabel("SalePrice")  # add Y-axis label 
plt.title("Sale Price by GDP Growth")  # add title 
plt.show()


# In[40]:


#Prices fell harshly during economic downturn as well, as is to be expected.


# In[41]:


#As for models, an attempt will be made at using machine learning to forecast the future trends in price, with a simple 
#training and testing set. 


# In[42]:


scaler = preprocessing.StandardScaler()
# Fitting data to the scaler object
scaled_df = scaler.fit_transform(Essential)
scaled_df = pd.DataFrame(scaled_df)


# In[43]:


# Subsetting our data into our dependent and independent variables.
Essential_train = Essential2[Essential2['YrSold'].isin([2006,2007,2008])]
X_train = Essential_train.drop(['SalePrice'],axis=1)
y_train = Essential_train['SalePrice']
Essential_test = Essential2[Essential2['YrSold'].isin([2009,2010])]
X_test= Essential_test.drop(['SalePrice'],axis=1)
y_test = Essential_test['SalePrice']


# In[44]:


#The first model to test for predictive power is the Random Forest Model, with a thorough examination of Bayesian possibility,
#this model may provide insights a more linear approach could miss.
#clf = RandomForestClassifier(n_estimators=300, random_state = 1,n_jobs=-1)
#X_train_scaled=scaler.fit_transform(X_train)
#X_test_scaled=scaler.fit_transform(X_test)
#model_res = clf.fit(X_train_scaled, y_train)
#y_pred = model_res.predict(X_train_scaled)
#y_pred_prob = model_res.predict_proba(X_train_scaled)
#lr_probs = y_pred_prob[:,1]
#ac = accuracy_score(y_train, y_pred)

#f1 = f1_score(y_test, y_pred, average='weighted')
#cm = confusion_matrix(y_test, y_pred)

#print('Random Forest: Accuracy=%.3f' % (ac))

#print('Random Forest: f1-score=%.3f' % (f1))


# In[45]:


from sklearn.ensemble import RandomForestRegressor
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[46]:


regressor = RandomForestRegressor(n_estimators=300, random_state=0, oob_score=True)
regressor.fit(X_train, y_train)


# In[47]:


oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')


# In[ ]:





# In[48]:


predictions = regressor.predict(X_test)
rmse(y_test, predictions)


# In[49]:


RFResults = regressor.fit(X_test, y_test)
ResultOobScore = RFResults.oob_score
print(f'Out-of-Bag Score: {ResultOobScore}')


# In[54]:


RFResults.feature_importances_


# In[50]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestClassifier(n_estimators=300, random_state = 1,n_jobs=-1)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
model_res = clf.fit(X_train_scaled, y_train)
y_pred = model_res.predict(X_test_scaled)
y_pred_prob = model_res.predict_proba(X_test_scaled)
lr_probs = y_pred_prob[:,1]
ac = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print('Random Forest: Accuracy=%.3f' % (ac))

print('Random Forest: f1-score=%.3f' % (f1))


# In[ ]:


regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
regressor


# In[ ]:


#The Random Forest Accuracy remains quite poor, only 80% of 1% accurate, and F1, tracking precision and recall is similarly
#low, reaching only 70% of 1%


# In[ ]:


from sklearn import linear_model, preprocessing 
rModel = linear_model.LinearRegression(normalize=True)


# In[ ]:


rModel.fit(X_train, y_train)


# In[ ]:



LinearRMSE = rmse(y_test, y_pred)
print(LinearRMSE)


# In[ ]:


import statsmodels.api as sm 
rModel2 = sm.OLS(y_train, X_train) 
rModel2_results = rModel2.fit()
rModel2_results.summary()


# In[ ]:


print(rModel.score(X_test, y_test))


# In[ ]:


#Strong results as well, but in now way divergent from the findings of the original model, the forecast finds a strong
#negative association with remodel recency, as expected, with a 180 times decrease in sale price for a remodel only one year
#out of date, implying a stronger negative relationship than the initial analysis was able to cover, and at a 95% Confidence 
#interval. Meanwhile, the new variables around GDP growth and inflation seem strong, with a strongly negative coefficient
#for the former, and a positive for the latter, but neither are statistically significant, implying the market is not
#especially affected by the changes in economic conditions in light of the 2008 financial crisis. Furthermore, the different
#model seems to be slightly less explanatory than the previous, non-forecast model, explaining 1% less variation.


# In[ ]:


#KNN Model:
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,10):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


print(test_scores)
print(train_scores)


# In[ ]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,10),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,10),test_scores,marker='o',label='Test Score')


# In[ ]:


knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
knn.score(X_train, y_train)


# In[ ]:


knn.fit(X_test, y_test)
knn.score(X_test, y_test)


# In[ ]:


#While a significant improvment over the Random Forest accuracy, it still lags behind the linear model, at a fit rate of only 
#22%


# In[ ]:





# In[ ]:




