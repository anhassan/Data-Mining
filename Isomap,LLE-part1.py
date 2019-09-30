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
data=pd.DataFrame(data[data['gnd']==3]) #removing the data with labels which are 3
print(data.head())

features = data.drop(columns=[list(data)[-1]]) #removing the target from the data to make features
target = data[[list(data)[-1]]] #keeping the target seperate
image_data=features.values
n_samples, n_features=image_data.shape,image_data.shape

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

def plot_embedding(X,image_data, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 3e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image_data[i,:].reshape(28,28),zoom=.45, cmap=plt.cm.gray_r),
                X[i],frameon=False)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.xlabel("Dimension 1",fontsize=12)
    plt.ylabel("Dimension 2",fontsize=12)
    if title is not None:
        plt.title(title,fontsize=14)


lle = LocallyLinearEmbedding(n_neighbors = 5,n_components=4)
lle_feats=lle.fit_transform(features)
LLE_14 = pd.DataFrame(data=lle_feats,columns=['lle1','lle2','lle3','lle4'])
LLE_12=LLE_14.drop(columns=['lle3','lle4']) #taking only the first dimensions for plotting



plot_embedding(LLE_12.values,image_data,"2 Dimensional Locally Linear Embedding with 5 Neighbors")
plt.show()


isomap = Isomap(n_neighbors=5,n_components = 4)
isomap_feats = isomap.fit_transform(features)
isomap_14 = pd.DataFrame(data=isomap_feats,columns=['iso1','iso2','iso3','iso4'])
isomap_12=isomap_14.drop(columns=['iso3','iso4'])


plot_embedding(isomap_12.values,image_data," 2 Dimensional Isomap with 5 Neighbors " )
plt.show()

