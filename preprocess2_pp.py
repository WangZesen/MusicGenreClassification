import pp, pickle, os
import numpy as np


fileLists = os.listdir("extractData")

for fileName in fileLists:
	if fileName.endswith(".pp"):
		curData = None
		with open("extractData/" + fileName, 'r') as f:
			content = f.read()
			curData = np.asarray(pickle.loads(content))
			curData = np.swapaxes(curData, 0, 2)
			curData = np.swapaxes(curData, 1, 2)
			curData = curData.tolist()
		with open("extractData/" + fileName, 'w') as f:
			f.write(pickle.dumps(curData))
		print fileName
