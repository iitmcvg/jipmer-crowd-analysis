# Jipmer-Crowd-Analysis
Developing intelligent crowd analysis pipeline for hospitals (JIPMER IITM collaboration project).


The paper we are primarily implementing is [resnet crowd](https://arxiv.org/pdf/1705.10698.pdf).

## Implementation details
Using `tensorflow-keras` for implementation.
Created `TfRecord files` for the input data and parsed using `Tensorflow Data-API`.
Used `ShanghaiTech dataset` for the training purpose.

#### Details of model
Did `multi-task learning` by truncating layers from `resnet50` by adding two branches of network. 
Performed `transfer learning` using `imagenet weights`, freezing the resnet50 part of the model and training only the added branches.
Using `AdaGrad Optimizer` and `L2 loss` for both
`heatmap` and `count` estimation.

## Results


## Roadmap
1. [x] Heatmaps from images.
2. [x] Count from images.
3. [ ] Violent activity recognition.
4. [ ] Crowd movement prediction.
5. [ ] Locating and tracking of abnormal detected region.
6. [ ] Person re-identification
