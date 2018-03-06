# MusicGenreClassification

IPP

# Feature

## Preprocess

* Mel-Frequency
* Log Power

![Alt text](analysis/mel/power.png?raw=true "Mel-Frequency Power")

![Alt text](analysis/mel/log_power.png?raw=true "Mel-Frequency Log Power")

* 5 channel better than 1 channel (preprocess_final.py)

## Model

* Different size of convolution kernel in different layers
* Concatenate last three layers with different sizes
* Two fully-connected layer (solve non-linear problem)

## Train

* Data setup (60% Train, 40% Validation, 10 Genres, etc.)
* SGD, 8 as batch size
* Layer-wise pretrain (http://www.iro.umontreal.ca/~lisa/pointeurs/BengioNips2006All.pdf)

## Post-process

* Extract feature from layer "fc1"
* Use Principle Component Analysis to reduce dimension of feature

## Experiment

* Accuracy, Top-3 accuracy, generalization error (compare with other paper, compare with 1 channel) 
  * https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
  
| Experiment | Accuracy | Top-3 Accurcay | Generalization Error | 5-NN Error |
|------------|----------|----------------|----------------------|------------|
| 5-channel Train  |   0.998   | | 0.000 | 0.997 |
| 5-channel Val | 0.725 | | 0.003 | 0.772 |
| 5-channel Test | 0.407 | | 0.000 | 0.79 |

* Result of PCA
* Cluster test on song lists of netease music (maybe)

## Conclusion

* ?

data file link: https://drive.google.com/drive/folders/1WQ0koI3FuHLLomt8itUAH0475ZfuN303?usp=sharing
