import os
import pickle
import numpy as np


count = 0
labels = []
data = []
numOfGenres = 10
genres = ["blues", "classic", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

fileLists = os.listdir("extractData")
for fileName in fileLists:
	if fileName.endswith(".pp"):
		with open("extractData/" + fileName, 'r') as f:
			content = f.read()
			curData = np.asarray(pickle.loads(content))
			curData = np.swapaxes(curData, 0, 2)
			curData = np.swapaxes(curData, 1, 2)
			# print curData.shape
			data.append(curData)
			for i in range(numOfGenres):
				if fileName.startswith(genres[i]):
					curLabel = [0.] * numOfGenres
					curLabel[i] = 1.
					labels.append(curLabel)
		count += 1
	if count > 50:
		break	


data = np.ndarray.tolist(np.array(data))
with open("data/data", "w") as f:
	f.write(pickle.dumps(data))
with open("data/label", "w") as f:
	f.write(pickle.dumps(labels))
			

'''
n_input = 599 * 128 * 5
with open("data", 'r') as f:
	content = f.read()
	data = np.asarray(pickle.loads(content))
data.reshape((data.shape[0], n_input))
print data.shape
'''
