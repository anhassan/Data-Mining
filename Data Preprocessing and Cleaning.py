import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt



df=pd.read_csv("C:/Users/HP/Downloads/DataA.csv",index_col=0)
print(df.info())
zz=sns.boxplot(x=df['fea.2'])
plt.title("Box Plot Before Outlier Removal",fontsize=15)
plt.show()

#three features namely:"fea.34", "fea.35", "fea.36", are almost filled with null. So they doesn't have any information.
#missing values in every features
#Data is not normalized.
#Outliners in features such as in fea.2 indicated in box plot

null_counts = df.isnull().sum()/len(df)
plt.figure(figsize=(20,6))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)


plt.title("Fraction of Missing Values")
plt.show()




df = df.drop(["fea.34", "fea.36", "fea.35"], axis=1) # Remove rows/features
                                                      #with almost no information
df=df.fillna(df.mean()) #Replace NA values by mean
                        #of column



z = np.abs(stats.zscore(df))

df_o = df[(z < 3).all(axis=1)] # removing outliners
                               #with zscore>3

zzz=sns.boxplot(x=df_o['fea.2'])# outliners removed such as in fea.2
plt.title("Box Plot After Removing Outliers")
plt.show()
df_z = (df_o - df_o.mean()) / df_o.std() #normalization with z-score
df_mm= (df_o - df_o.min()) / (df_o.max() - df_o.min())#normalization with min max



sns.set()
_=plt.hist(df_o['fea.9'])
plt.title("Without Normalization Fea.9")
plt.show()

_=plt.hist(df_z['fea.9'])
plt.title("With z-score Normalization Fea.9")
plt.show()

_=plt.hist(df_mm['fea.9'])
plt.title("With min-max Normalization Fea.9")
plt.show()


_=plt.hist(df_o['fea.24'])
plt.title("Without Normalization Fea.24")
plt.show()
_=plt.hist(df_z['fea.24'])
plt.title("With z-score Normalization Fea.24")
plt.show()
_=plt.hist(df_mm['fea.24'])
plt.title("With min-max Normalization Fea.24")
plt.show()

# Before normalization the data values varies to greater extend but after normalization the
#values are now bounded e.g between -3 and 3 in case of z-score
#and between 0 and 1 in case of min-max