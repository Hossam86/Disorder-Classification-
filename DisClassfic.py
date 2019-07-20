from  sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff

import pandas as pd
import matplotlib.pyplot as plt

data=arff.loadarff('DataSets\Autism-Adolescent-Data.arff')
df=pd.DataFrame(data[0])

# df = df.select_dtypes([np.object]) 
df = df.stack().str.decode('utf-8').unstack()

#data preprocessing - transform catagorial data
df=df.apply(lambda x:x.astype(str).str.lower())
df=df.replace('yes',1)
df=df.replace('no',0)
df=df.replace('f',1)
df=df.replace('m',0)

# subset the data
xVar=list(df.loc[:,'A1_Score':'A10_Score'])+['gender']+['jundice']+['austim']
yVar=df.iloc[:,20]
df2=df[xVar]
print(df2.head())

# Split the Data into Train and Test Sets 

X_train,X_test,y_train,y_test=train_test_split(df2,yVar,test_size=0.2)

# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)

# Build A random Forest 

clf=RandomForestClassifier(n_estimators=10,n_jobs=-1, random_state=0)
clf.fit(X_train,y_train)

# Predict
preds=clf.predict(X_test)

print (pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))

print(list(zip(X_train, clf.feature_importances_)))
