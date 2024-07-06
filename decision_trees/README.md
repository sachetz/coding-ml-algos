# Decision Trees

The decision tree algorithm is implemented in Python for the arrhythmia dataset.

Each attribute in the dataset is either a binary discrete variable or a continuous variable. The algorithm builds a decision tree based on the training data, where each internal node represents a test on the attribute that has the highest information gain, and generates a binary split for the node.

## Calculation of information gain threshold

1. For continuous attributes, this binary split is generated based on values relative to a threshold. To calculate the possible threshold values, the data is first sorted based on that attribute, and all possible average values between two consecutive data points with different classes are considered. The final threshold is selected to be the one that provides the highest information gain across all possible threshold values for that attribute for the subset of the data at that node.
2. For discrete binary attributes, this threshold is taken to be the midpoint of the two values. Doing so provides code simplicity, as these discrete attributes can be treated as continuous attributes without loss of functionality.

## Validation

The validation curve is built using 3-fold cross validation. The dataset is split into three parts, and trained thrice using different pairs of these parts, with the test conducted on the third pair on each training iteration. The training is done across different tree depths ranging between 2 and 16 with increments of 2, and the average train vs test accuracy is plotted at each of these steps in the validation curve.

## Observations

We notice that as the depth of the tree is increased, we gain a higher training accuracy (which reaches 100% for depth 14-16), however the best test accuracy is obtained at the depth of 4. This suggests that the model overfits the data for higher values of tree depth, and an ideal approach would involve the use of tree of depth 4-6. The maximum test accuracy obtained is around 67.5%, which could be improved by the use of different model, or with a larger dataset.