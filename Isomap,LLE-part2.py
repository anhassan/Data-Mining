import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler
from matplotlib import offsetbox
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA
from numpy.linalg import norm
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from time import time
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

data = pd.read_csv("C:/Users/HP/Downloads/DataB.csv")


data=data.drop(columns=[list(data)[0]]) #dropping the first column because it was indices only

features = data.drop(columns=[list(data)[-1]]) #removing the target from the data to make features
target = data[[list(data)[-1]]] #keeping the target seperate


scaler = MinMaxScaler()
features = scaler.fit_transform(features)


def score_lle(x_train,y_train,x_test,y_test):
    lle = LocallyLinearEmbedding(n_neighbors=5, n_components=4)
    x_train=lle.fit_transform(x_train)
    x_test=lle.fit_transform(x_test)
    nb=GaussianNB()
    nb.fit(x_train,y_train)
    y_pred=nb.predict(x_test)
    return accuracy_score(y_pred,y_test)


def score_isomap(x_train,y_train,x_test,y_test):
    iso = Isomap(n_neighbors=5, n_components=4)
    x_train=iso.fit_transform(x_train)
    x_test=iso.fit_transform(x_test)
    nb=GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    return accuracy_score(y_pred, y_test)

def score_pca(x_train,y_train,x_test,y_test):
    pca = PCA(n_components=4)
    x_train=pca.fit_transform(x_train)
    x_test=pca.fit_transform(x_test)
    nb=GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    return accuracy_score(y_pred, y_test)

def score_lda(x_train,y_train,x_test,y_test):
    lda1 = LinearDiscriminantAnalysis(n_components=4)
    lda2 = LinearDiscriminantAnalysis(n_components=4)
    lda1.fit(x_train,y_train)
    x_train=lda1.transform(x_train)
    lda2.fit(x_test,y_test)
    x_test=lda2.transform(x_test)
    nb=GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    return accuracy_score(y_pred, y_test)

scores_lle=[]
scores_isomap=[]
scores_pca=[]
scores_lda=[]
iterations=[]
for i in range(2,11):
    print("here = ",i)
    iterations.append(i)
    cv = ShuffleSplit(n_splits=i, test_size=0.30, random_state=0)
    scores_1 = []
    scores_2 = []
    scores_3 = []
    scores_4 = []
    for train_index, test_index in cv.split(features):
        x_train = features[train_index]
        x_test = features[test_index]
        target = np.array(target)
        y_train = target[train_index]
        y_test = target[test_index]
        scores_1.append(score_lle(x_train,y_train,x_test,y_test))
        scores_2.append(score_isomap(x_train, y_train, x_test, y_test))
        scores_3.append(score_pca(x_train, y_train, x_test, y_test))
        scores_4.append(score_lda(x_train, y_train, x_test, y_test))
    scores_lle.append(np.mean(np.array(scores_1)))
    scores_isomap.append(np.mean(np.array(scores_2)))
    scores_pca.append(np.mean(np.array(scores_3)))
    scores_lda.append(np.mean(np.array(scores_4)))

print("lle = ",scores_lle)
print("isomap = ",scores_isomap)
print("pca = ",scores_pca)
print("lda = ",scores_lda)
print("iterations = ",iterations)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xticks([]),ax.set_yticks([])
ax.set_xlabel('Number of Iterations', fontsize = 15)
ax.set_ylabel('Accuracy', fontsize = 15)
ax.set_title('Locally Linear Embedding', fontsize = 20)


ax.plot(iterations,scores_lle)
ax.grid()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xticks([]),ax.set_yticks([])
ax.set_xlabel('Number of Iterations', fontsize = 15)
ax.set_ylabel('Accuracy', fontsize = 15)
ax.set_title('Isomap', fontsize = 20)

ax.plot(iterations,scores_isomap)
ax.grid()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xticks([]),ax.set_yticks([])
ax.set_xlabel('Number of Iterations', fontsize = 15)
ax.set_ylabel('Accuracy', fontsize = 15)
ax.set_title('Principal Component Analysis', fontsize = 20)

ax.plot(iterations,scores_pca)
ax.grid()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xticks([]),ax.set_yticks([])
ax.set_xlabel('Number of Iterations', fontsize = 15)
ax.set_ylabel('Accuracy', fontsize = 15)
ax.set_title('Linear Discriminant Analysis', fontsize = 20)

ax.plot(iterations,scores_lda)
ax.grid()
plt.show()


