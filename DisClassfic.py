from  sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff

import pandas as pd

data=arff.loadarff('DataSets\Autism-Adolescent-Data.arff')
df=pd.DataFrame(data[0])
list(df)

#data preprocessing - transform catagorial data
df=df.apply(lambda x:x.astype(str).str.lower())
df.replace('yes',1)
df.replace('no',0)
df.replace('f',1)
df.replace('m',0)

# subset the data
xVar=list(df.loc[:,'A1_Score':'A10_Score'])+['gender']+['jundice']+['austim']
