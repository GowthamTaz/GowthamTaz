import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'x':[12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,61,64,69,72,24,15,6],
    'y':[39,36,30,52,54,45,55,59,46,55,59,63,70,66,63,58,23,14,8,19,7,24]
})
print(len(df['x']))
print(len(df['y']))
K_means = KMeans(n_clusters=4)
K_means.fit(df)

Label = K_means.predict(df)
centroids = K_means.cluster_centers_ # some of squared error in the perticular cluster is called inertia_
print(centroids)

Color = {1:'y',2:'b',3:'g'}
Col_map =list(map(lambda x:Color[x+1],Label))
# plt.subplot(3,1,1)
plt.scatter(df['x'],df['y'],color = Col_map,alpha = 0.5,edgecolor = 'k')
plt.show()
for idx, centroid in enumerate(centroids):
    plt.subplot(3,1,2)
    plt.scatter(*centroid,color = Col_map[idx+1])

plt.xlim(-25,100)
plt.ylim(-10,125)
plt.show()
New = np.array([25,45]).reshape(1,-1)
print(list(New))

Test = K_means.predict(New)
Test_1 = np.array([90,45]).reshape(1,-1)
Test_2 =  K_means.predict(Test_1)
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
#Visualizing the ELBOW method to get the optimal value of K
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()
