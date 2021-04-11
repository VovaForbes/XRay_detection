# XRay_detection

Current solution does not achieve SOTA results. Current best evaluation metric is 0.22 mAP (all classes treated as one and the result is measured only on first fold, since 5-fold cross-val was tooo time-consuming). IOUs are (0.25, 0.5, 0.05).

The reasons of this are that the dataset was very small despite the fact it was expanded with 3.3k images from VinBigData chest XRay kaggle competition (https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection). Classes Mass/Nodule were put together to match the mark in the kaggle dataset.

Since the purpose of pipeline was not to achieve SOTA results, there are some work to be done to solve this issue:
1) Pseudo labeling on unmarked images.
2) Stacking several models (learnt on different folds). (https://arxiv.org/pdf/1910.13302.pdf)
3) Trying custom augmentations such as mosaic augmentation and so on. (https://arxiv.org/pdf/1906.11172.pdf)
4) Writing custom sampler to deal with class imbalance. (https://arxiv.org/pdf/1802.05033.pdf, https://ecmlpkdd2019.org/downloads/paper/624.pdf)
5) Do label smoothing since the provided initial mark was not checked by the radiologists, but scrapped among several sources.
6) Postprocessing such as detecting distribution of bounding boxes among image coordinates to reduce number of false positives.
7) Detect correlation between occurancies of different types of abnormalties. (To correct labels or add some aprior probabilities to predictions).
8) Try different backbones.
9) Surf internet to find even more train data. (Since before adding 3.3k images from kaggle the mAP metric didn't exceed 0.1).
10) Also it is a good idea to add Stratified k-fold:) instead of just k-fold.
