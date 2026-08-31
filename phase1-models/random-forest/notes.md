# Random Forest Research Notes

## Summary

A random forest is a collection of individual decision trees working together to produce a result. Each tree must be diverse to reduce unwanted noise being included in the decision making process.

## Decision Trees

Decision trees work by breaking down complex problems into a series of yes or no questions to produce an answer. It consists of nodes where the data is split based on a rule. The split data then follows its respective branch to another node, where the data is aplit again. This continues until a final answer is produced.

Decision trees are made by evaluating which input features split the input into the correct target classes.

For each possible split, it measures it appropriateness based on one of two mathematical metrics: Geni Impurity or Entropy.

### Gini Impurity

$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$

$$p_i = \frac{\text{Number of items in class } i}{\text{Total number of items in the node}}$$

Note: Minimum impurity is 0.0 and maximum is 0.5.

The Gini Impurity is used to reduce the diversity in the split data. Pure data is a set of data with a low Gini Impurity score, meaning that all data points belong to a single category or class. Gini Impurity aims to split data into the purest groups possible.

The process includes evaluating the original impurity of the set. The algorithm then tests every possible feature and rule. It then calculates which rule creates the most pure split by calculating the overall impurity of the two children nodes and comparing that impurity to the parents impurity.

To calculate the weighted childrens impurity:

$$Gini_{split} = \frac{N_{left}}{N_{total}} Gini_{left} + \frac{N_{right}}{N_{total}} Gini_{right}$$

### Entropy and Information Gain

$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

The process to deciding which rule to use is exactly the same for Entropy as it is Gini Impurity, but with using the Entropy formula rather than Gini Impurity formula.

The minimum value is 0.0 and the maximum is 1.0.

Gini Impurity is the preffered option in a realistic setting. This is due to it having a faster computational speed with no performance difference between the two methods.

## Diverse Trees in a Random Forest

There are two major techniques used to ensure diversity among trees in a random forest: Bootstrap Aggregation and Feature Randomness.

### Bootstrap Aggregation

Each decision tree is made using input data. In order to ensure diversity among a random forest, each decision tree is built on a random set of data from the entire training sets data. Objects are randomly selected from the training data, with replacement, before being used to build each tree.

### Feature Randomness

Rather than evaluate each tree on all features, a small subset of features are used to build each tree. The size of the subset is usually $\sqrt{\text{total features}}$. This prevents single highly predictive features dominating each tree, causing them to be near identical.

Random forests work by running the test data through each individual decision tree. In the case of a clasification, a majority vote is used to decide the overall output. For regression, a mean is calculated and outputted.

## Hyperparameters

Hyperparameters effect a models performance by balancing overfitting and underfitting. They are set manually before training and include things like maximum depth of a tree, number of decision trees in the forest and other important numbers that need to be manually decided on before training begins. They can drastically effect the outcome of the random forest.

There are multiple methods to find the optimal hyperparameters for a data set, including grid search and random search, which both search the possible combinations of a hyperparameters for the optimal set.

## Out-Of-Bag Score

In order to build each tree, approxiamately ~67% of data in the data set is used. Therefore, the remaining ~33% of data in the data set is able to be used to test it. This means that you do not need a secondary testing set. Each data point not used to build the tree can then be used to evaluate it, using the same metrics as a logistic regression model.

## Practical Use

Random Forests dominate in areas where features have a non-linear or interdependent relationships with the classification. They are seen as a much more robust and resilient form of machine learning, but also require a lot more time to train.
