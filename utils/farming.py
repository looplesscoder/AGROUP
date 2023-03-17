import pandas as pd
import numpy as np
import seaborn as sns
import pickle 


# In[2]:


df= pd.read_csv(r"E:\Crop Prediction\Crop_recommendation.csv")


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.corr()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df['label'].value_counts()


# In[10]:


sns.heatmap(df.corr(), annot=True)


# In[11]:


features= df.drop(columns=['label'] ,axis=1)
target= df['label']


# In[12]:


features.head(10)


# SPLITTING THE DATASET

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test= train_test_split(features, target, test_size=0.3, random_state=1)


# MODEL SELECTION

# In[15]:


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')


# In[16]:


acc=[]
models=[]


# Logistic Regression

# In[17]:


LR= LogisticRegression(random_state=3 )
LR.fit(x_train,y_train)
predicted= LR.predict(x_test)
val = metrics.accuracy_score(predicted ,y_test)
acc.append(val)
models.append('Logistic_Regression')
print("Logistic Regression Accuracy is: ", val)
print(classification_report(y_test,predicted))


# In[18]:


score = cross_val_score(LR,features,target,cv=5)
score


# Random Forest

# In[19]:


RF = RandomForestClassifier(n_estimators=42, random_state=0)
RF.fit(x_train,y_train)

predicted= RF.predict(x_test)

val = metrics.accuracy_score(y_test, predicted)
acc.append(val)
models.append('RandomForest')
print("RF's Accuracy is: ", val)

print(classification_report(y_test,predicted))


# In[20]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5,scoring='f1_macro')
score


# In[21]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))


# In[38]:


#open the file to save as pkl file
pickle.dump(RF, open('model.pkl', 'wb'))
model= pickle.load(open('model.pkl','rb'))
# RF_model_pkl.close()


# SVM Classifier

# In[22]:


svc_model = SVC(C= .1, kernel='linear', gamma= 1)
svc_model.fit(x_train, y_train)
 
prediction = svc_model .predict(x_test)
# check the accuracy on the training set
val = metrics.accuracy_score(y_test, predicted)
print("svm's Accuracy is: ", val)
acc.append(val)
models.append('SVMclassifier')

print(classification_report(y_test,predicted))


# In[23]:


SVM = SVC(kernel='rbf')
SVM.fit(x_train,y_train)

predicted= SVM.predict(x_test)

val = metrics.accuracy_score(y_test, predicted)
acc.append(val)
models.append('SVMwithrbf')
print("svm's Accuracy is: ", val)

print(classification_report(y_test,predicted))


# In[24]:


score = cross_val_score(svc_model,features,target,cv=5)
score


# Naive Bayes

# In[26]:


NaiveBayes = GaussianNB()
NaiveBayes.fit(x_train,y_train)
predicted = NaiveBayes.predict(x_test)
val = metrics.accuracy_score(y_test, predicted)
acc.append(val)
models.append('Naive_Bayes')
print("Naive Bayes's Accuracy is: ", val)
print(classification_report(y_test,predicted))


# In[27]:


score = cross_val_score(NaiveBayes,features,target,cv=5)
score


# DecisionTreeClassifer

# In[28]:


DecisionTree= DecisionTreeClassifier(criterion= 'entropy' , random_state=3 ,max_depth=9)
DecisionTree.fit(x_train,y_train)
predicted= DecisionTree.predict(x_test)
val = metrics.accuracy_score(predicted ,y_test)
acc.append(val)
models.append('DecisionTree')
print("DecisionTrees's Accuracy is: ", val)
print(classification_report(y_test,predicted))


# In[29]:


score = cross_val_score(DecisionTree,features,target,cv=5)
score


# CHECKING BEST ACCURACY

# In[30]:


import matplotlib.pyplot as plt 


# In[35]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = models,y =acc, palette= 'dark')


# In[34]:


accuracy_models = dict(zip(models, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# In[ ]:




