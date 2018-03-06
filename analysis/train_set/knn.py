import numpy as np

f1 = open("results_feature.txt", "r")
f2 = open("results_label.txt", "r")

k = 10
n = 120 * 8

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
print features[0]
print label[0:50]

error_count = 0.

for i in range(n):
	dists = []
	for j in range(n):
		dists.append(np.sum(np.square(features[i] - features[j])))
	args = np.argsort(np.array(dists))
	label_count = [0 for j in range(10)]
	for j in range(k):
		label_count[int(label[args[j]])] += 1
	if label[i] != np.argmax(label_count):
		error_count += 1
		
print error_count / n


f1.close()
f2.close()
