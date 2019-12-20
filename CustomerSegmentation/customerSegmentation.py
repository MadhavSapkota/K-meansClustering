#*********************CUSTOMER SEGMENTATION WITH K-MEANS******************************

#1.****************************DATA DOWNLOAD********************* 

	!wget -O Cust_Segmentation.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv

# 2. **************LOADING DATA FROM CSV FILE**********
	import pandas as pd
	cust_df = pd.read_csv("Cust_Segmentation.csv")
	cust_df.head()

#3. *********************PREPROCESSING*******************
	df = cust_df.drop('Address', axis=1)
	df.head()


#******Normalizing over the standard deviation************* 
	from sklearn.preprocessing import StandardScaler
	X = df.values[:,1:]
	X = np.nan_to_num(X)
	Clus_dataSet = StandardScaler().fit_transform(X)
	Clus_dataSet


# 4. *****************MODELLING***********
# Applying k-means

	clusterNum = 3
	k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
	k_means.fit(X)
	labels = k_means.labels_
    print(labels)


#5. ****************INSIGHT********
	df["Clus_km"] = labels
	df.head(5)

	df.groupby('Clus_km').mean()

#--Distribution of customers based on age and income
	area = np.pi * ( X[:, 1])**2  
	plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
	plt.xlabel('Age', fontsize=18)
	plt.ylabel('Income', fontsize=16)
	plt.show()
	

	from mpl_toolkits.mplot3d import Axes3D 
	fig = plt.figure(1, figsize=(8, 6))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	plt.cla()
	# plt.ylabel('Age', fontsize=18)
	# plt.xlabel('Income', fontsize=16)
	# plt.zlabel('Education', fontsize=16)
	ax.set_xlabel('Education')
	ax.set_ylabel('Age')
	ax.set_zlabel('Income')

	ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

	


