from  sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff

import pandas as pd

data=arff.loadarff('DataSets\Autism-Adolescent-Data.arff')
df=pd.DataFrame(data[0])

#data preprocessing - transform catagorial data
df=df.apply(lambda x:x.astype(str).str.lower())
df.replace('yes',1)
df.replace('no',0)
df.replace('f',1)
df.replace('m',0)

# subset the data
xVar=list(df.loc[:,'A1_Score':'A10_Score'])+['gender']+['jundice']+['austim']
yVar=df.iloc[:,20]
df2=df[xVar]


# Split the Data into Train and Test Sets 

X_train,X_test,y_train,y_test=train_test_split(df2,yVar,test_size=0.2)

# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)
