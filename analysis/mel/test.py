import numpy as np
import pickle
import librosa
import librosa.display as display
import matplotlib.pyplot as plt

f = open("blues.00000.pp", "r")
content = f.read()
a = np.asarray(pickle.loads(content))


c = a[0]
b = librosa.power_to_db(a[0].T, ref = np.max)
# b = a[0].T

b = (b - np.min(b)) / (np.max(b) - np.min(b))
print b.shape
print np.max(b)

ax = display.specshow(b)
ax.axis("tight")
ax.axis("off")
plt.savefig("test.png", transparent = True, bbox_inches = "tight", pad_inches = 0)
