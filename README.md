# MGDTI
code implementation for MGDTI


PULLED DATA FROM MOLTRANS DAVIS



The core problem I've encountered lies in the need for features from certain data points to be part of the model's feature list from the very beginning, while also excluding the target values for these points from the training process. 

The current code does not support this approach cleanly, as it assumes that all features are linked to their corresponding target labels, leaving no straightforward way to create a test set that remains hidden during training. Without an efficient mechanism for masking out these specific targets while keeping the features in play, the model risks implicit data leakage—where the training process could unintentionally learn from information meant to be kept out of the training set. This leakage undermines the fairness of the evaluation, as it may artificially inflate performance, leading to misleading results when testing the model on truly unseen data.