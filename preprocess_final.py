import matplotlib.pyplot as plt
from sys import argv
import sys
import numpy
import librosa
import pickle

import os
import pp
import time

def halt_with_usage():
	# """ HELP MENU """
	print 'USAGE: python preproccess_final.py [path to audio data] [data type (mp3, au, etc.)]'
	sys.exit(0)

def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)
	

def preprocess_audio(audioPath, ppFilePath):

	SOUND_SAMPLE_LENGTH = 30000
	HAMMING_SIZE = 100
	HAMMING_STRIDE = 40

	

	featuresArray = []
	for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
		if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:
			y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)

			# Let's make and display a mel-scaled power (energy-squared) spectrogram
			S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

			# Convert to log scale (dB). We'll use the peak power as reference.
			# log_S = librosa.power_to_db(S, ref = np.max)

			# mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
			# featuresArray.append(mfcc)

			# librosa.display.specshow(log_S, y_axis = "mel", x_axis = "time")

			featuresArray.append(S)

			if len(featuresArray) == 599:
				break
	featuresArray = numpy.array(featuresArray)
	featuresArray = numpy.swapaxes(featuresArray, 0, 2)
	for i in range(5):
		featuresArray[i] = librosa.power_to_db(featuresArray[i], ref = numpy.max)
	featuresArray = numpy.swapaxes(featuresArray, 1, 2)
	
	# print featuresArray.shape
	# print 'storing pp file: ' + ppFilePath

	f = open(ppFilePath, 'w')
	f.write(pickle.dumps(featuresArray))
	f.close()
	
	print 'Processed', audioPath

if __name__ == "__main__":

	if len(argv) != 3:
		halt_with_usage()
	
	job_server = pp.Server(7, ppservers = ())
	jobs = []
	
	walk_dir = argv[1]
	storage_dir = walk_dir + "_extract_data"
	audio_type = argv[2]
	
	print "walk dir =", walk_dir
	print "storage dir =", storage_dir
	
	if not os.path.isdir(storage_dir):
		os.mkdir(storage_dir)
	
	deprecated_genre = ["Electric", "Folk"]
	
	for root, subdirs, files in os.walk(walk_dir):
		for file_name in files:
			if file_name.endswith(audio_type):
				quit = False
				for deprecated in deprecated_genre:
					quit = quit or file_name.startswith(deprecated)
				if quit:
					continue
				ppFileName = storage_dir + "/" + file_name
				ppFileName = rreplace(ppFileName,  audio_type, "pp", 1)
				filePath = os.path.join(root, file_name)
				try:
					jobs.append(job_server.submit(preprocess_audio, (filePath, ppFileName,), (), ("librosa", "numpy")))
				except Exception as e:
					print "Error Occured", str(e)
	for i in range(len(jobs)):
		jobs[i]()
	'''
	start_time = time.time()
	preprocess_audio("testdata/Classic/1.mp3", "testdata/1.pp")
	print "{} seconds".format(time.time() - start_time)
	'''



