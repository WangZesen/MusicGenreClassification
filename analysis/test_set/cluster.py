import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f1 = open("results_feature.txt", "r")
f2 = open("results_label.txt", "r")

n = 50 * 8

features = np.zeros((n, 50))
label = np.zeros((n,))

feature_index = 0
label_index = 0

auxil_count = 0

feature_lines = f1.readlines()
label_lines = f2.readlines()

for i in range(len(feature_lines)):
	line = feature_lines[i].split(" ")
	for j in range(len(line)):

		if line[j].endswith("]]\n"):
			line[j] = line[j][:-3]
			auxil_count += 1
		elif line[j].endswith("]\n"):
			line[j] = line[j][:-2]			
		elif line[j].startswith("[["):
			line[j] = line[j][2:100]
		elif line[j].startswith("["):
			line[j] = line[j][1:100]
		try:
			features[feature_index // 50][feature_index % 50] = float(line[j])
			feature_index += 1
		except:
			pass

for i in range(len(label_lines)):
	line = label_lines[i].split(" ")
	for j in range(len(line)):
		if line[j].endswith("]\n"):
			line[j] = line[j][:-2]
		try:
			label[label_index] = float(line[j])
			label_index += 1
		except:
			if len(line[j]) > 2:
				print line[j]
			pass	
# test_case = 6400 - 1
# print features[test_case // 50][test_case % 50]

print label_index
print feature_index

f1.close()
f2.close()


pca = PCA(n_components = 3)
pca.fit(features)
positions = pca.transform(features)

colors = ["black", "rosybrown", "red", "sienna", "gold", "chartreuse", "darkgreen", "plum", "aqua", "midnightblue"]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
for i in range(len(positions)):
    ax.scatter(positions[i][0], positions[i][1], positions[i][2], marker = "o", color = colors[int(label[i])], s = 4)
plt.show()


pca = PCA(n_components = 2)
pca.fit(features)
positions = pca.transform(features)

colors = ["black", "rosybrown", "red", "sienna", "gold", "chartreuse", "darkgreen", "plum", "aqua", "midnightblue"]

fig = plt.figure()

for i in range(len(positions)):
    plt.plot([positions[i][0]], [positions[i][1]], marker = "o", color = colors[int(label[i])])
plt.show()


