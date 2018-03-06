# MusicGenreClassification

IPP

# Feature

## Preprocess

* Mel-Frequency
* Log Power

![Alt text](analysis/mel/log_power.png?raw=true "Title")

![Alt text](analysis/mel/power.png?raw=true "Title")

* 5 channel

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

* Accuracy, Top-3 accuracy, generalization error (compare with other paper)
* Result of PCA
* Cluster test on song lists of netease music (maybe)

## Conclusion

* ?

data file link: https://drive.google.com/drive/folders/1WQ0koI3FuHLLomt8itUAH0475ZfuN303?usp=sharing
