import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import pp
import librosa

def die_with_usage():
	""" HELP MENU """
	print 'USAGE: python preproccess.py [path to MSD mp3 data]'
	sys.exit(0)

def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)


def prepossessingAudio(audioPath, ppFilePath):

	SOUND_SAMPLE_LENGTH = 30000
	HAMMING_SIZE = 100
	HAMMING_STRIDE = 40

	print 'Prepossessing ' + audioPath

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
	print audioPath
	print len(featuresArray)
	print featuresArray[0].shape

	print 'storing pp file: ' + ppFilePath

	f = open(ppFilePath, 'w')
	f.write(pickle.dumps(featuresArray))
	f.close()


if __name__ == "__main__":

	# help menu
	if len(sys.argv) < 2:
		die_with_usage()
	
	job_server = pp.Server(6, ppservers = ())
	jobs = []

	i = 0.0
	walk_dir = sys.argv[1]

	print('walk_dir = ' + walk_dir)

	for root, subdirs, files in os.walk(walk_dir):
		for filename in files:
			if filename.endswith('.au'):
				ppFileName = "data/" + filename
				file_path = os.path.join(root, filename)
				ppFileName = rreplace(ppFileName, ".au", ".pp", 1)

				try:
					jobs.append(job_server.submit(prepossessingAudio, (file_path, ppFileName,), (), ("librosa",)))
				except Exception as e:
					print "Error accured" + str(e)

			if filename.endswith('au'):
				sys.stdout.write("\r%d%%" % int(i / 7620 * 100))
				sys.stdout.flush()
				i += 1


	for i in range(len(jobs)):
		jobs[i]();
