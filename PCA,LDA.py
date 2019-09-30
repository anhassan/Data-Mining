import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv("C:/Users/HP/Downloads/DataB.csv")

data=data.drop(columns=[list(data)[0]]) #dropping the first column because it was indices only
target = data[[list(data)[-1]]] #keeping the target seperate
features = data.drop(columns=[list(data)[-1]]) #removing the target from the data to make features

X_sd = StandardScaler().fit_transform(features) # centralizing the data by removing mean
eigen_vec,eigen_val,trans_eigen_vec = svd(X_sd) #computing eigen vectors and eigen values
print("eigen_vectors:")
print(eigen_vec)
print("eigen_values :")
print(eigen_val)



scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#Applying PCA with two components
pca = PCA(n_components=2)
PCs12 = pca.fit(features)
print("PCA12 variances = ",pca.explained_variance_ratio_)
PCs12=pca.transform(features)
PCs12 = pd.DataFrame(data=PCs12, columns=['pc1','pc2'])

PCs12_target = pd.concat([PCs12,target],axis=1)

#Plotting the results
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

labels=[0,1,2,3,4]
colors =['r','g','b','y','k']

for label,color in zip(labels,colors):
    keep_index = PCs12_target['gnd']==label
    ax.scatter(PCs12_target.loc[keep_index,'pc1'],PCs12_target.loc[keep_index,'pc2'],c=color,s=50)
ax.legend(labels)
ax.grid()
#plt.show()

# Applying PCA with six components
pca = PCA(n_components=6)
PCs1to6 = pca.fit(features)
print("PCA16 variances = ",pca.explained_variance_ratio_)
PCs1to6 = pca.transform(features)
PCs1to6 = pd.DataFrame(data=PCs1to6, columns=['pc1','pc2','pc3','pc4','pc5','pc6'])

PCs1to6_target = pd.concat([PCs1to6,target],axis=1)
PCs56_target=PCs1to6_target.drop(columns=['pc1','pc2','pc3','pc4'])


#Plotting the results
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 5', fontsize = 15)
ax.set_ylabel('Principal Component 6', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

labels=[0,1,2,3,4]
colors =['r','g','b','y','k']

for label,color in zip(labels,colors):
    keep_index = PCs56_target['gnd']==label
    ax.scatter(PCs56_target.loc[keep_index,'pc5'],PCs56_target.loc[keep_index,'pc6'],c=color,s=50)
ax.legend(labels)
ax.grid()
#plt.show()

nc =[2,4,6,10,30,60,200,500,784]
list_accuracy=[]
list_var=[]
list_error=[]
#Training the Naive Bayes Classifier
for i in range(0,len(nc)):
    pca = PCA(n_components=nc[i])
    print(nc[i])
    pca.fit(features)
    list_var.append(np.sum(pca.explained_variance_ratio_))#computing the variances combined
    x=pca.transform(features)
    nb=GaussianNB()
    nb.fit(x,target.values)
    y_pred=nb.predict(x)
    list_accuracy.append(accuracy_score(target.values, y_pred))#computing the accuracy
    list_error.append(1-accuracy_score(target.values, y_pred))#computing the error

print("list_of_errors = ", list_error)
print("list_of_variances = ",list_var)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Total Variance', fontsize = 15)
ax.set_ylabel('Classification Error', fontsize = 15)
ax.set_title('Error vs Variance', fontsize = 20)

ax.plot(list_var,list_error)
ax.grid()
#plt.show()

#Applying LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(features,target)
lda_feats = lda.transform(features)
LDA_12 = pd.DataFrame(data=lda_feats, columns=['lda1','lda2'])
LDA_12_target = pd.concat([LDA_12,target],axis=1)
#Plotting results
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('LDA Component 1', fontsize = 15)
ax.set_ylabel('LDA Component 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)

labels=[0,1,2,3,4]
colors =['r','g','b','y','k']

for label,color in zip(labels,colors):
    keep_index = LDA_12_target['gnd']==label
    ax.scatter(LDA_12_target.loc[keep_index,'lda1'],LDA_12_target.loc[keep_index,'lda2'],c=color,s=50)
ax.legend(labels)
ax.grid()
plt.show()
