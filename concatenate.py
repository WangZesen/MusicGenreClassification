import os, sys, pickle
import numpy as np


count = 0
labels = []
data = []
numOfGenres = 10
genres = ["blues", "classic", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

fileLists = os.listdir("extractData")

train = False

for fileName in fileLists:
	if fileName.endswith(".pp"):
		with open("extractData/" + fileName, 'r') as f:
			content = f.read()
			curData = np.asarray(pickle.loads(content))
			curData = np.swapaxes(curData, 0, 2)
			curData = np.swapaxes(curData, 1, 2)
			data.append(curData[0:1])
			print fileName
			for i in range(numOfGenres):
				if fileName.startswith(genres[i]):
					curLabel = i
					labels.append(curLabel)
		count += 1
		
	if count > 100 and (not train):
		print "Writing Train Data to Disk..."
		
		listData = np.ndarray.tolist(np.array(data))
		with open("data/trainData", "w") as f:
			f.write(pickle.dumps(listData))
		with open("data/trainLabel", "w") as f:
			f.write(pickle.dumps(labels))
		
		# print labels
		# sys.exit(0)
		data = []
		labels = []
		train = True
		print "Finished"
		
	if count > 150:
		print "Writing Validation Data to Disk..."
		listData = np.ndarray.tolist(np.array(data))
		with open("data/valData", "w") as f:
			f.write(pickle.dumps(listData))
		with open("data/valLabel", "w") as f:
			f.write(pickle.dumps(labels))
		print "Finished"
		break
		

