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

* Data setup (60% Train, 40% Test, 10 Genres, etc.)
* Layer-wise pretrain

## Post-process

* Extract feature from layer "fc1"
* Use Principle Component Analysis to reduce dimension of feature

## Experiment

* Accuracy, Top-3 accuracy, generalization error (compare with other paper, compare with 1 channel)
* Result of PCA
* Cluster test on song lists of netease music (maybe)

## Conclusion

* ?

data file link: https://drive.google.com/drive/folders/1WQ0koI3FuHLLomt8itUAH0475ZfuN303?usp=sharing
