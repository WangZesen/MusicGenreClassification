import numpy as np

f1 = open("results_output.txt", "r")
f2 = open("results_label.txt", "r")

n = 120 * 8

output = np.zeros((n,))
label = np.zeros((n,))

output_index = 0
label_index = 0



output_lines = f1.readlines()
label_lines = f2.readlines()

for i in range(len(output_lines)):
	line = output_lines[i].split(" ")
	for j in range(len(line)):
		try:
			output[output_index] = float(line[j])
			output_index += 1
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

error_count = 0.

for i in range(0, n):
	if output[i] != label[i]:
		error_count += 1
print error_count
print error_count / n


f1.close()
f2.close()
