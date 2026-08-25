# Logistic Regression Research Notes

# Summary

Logistic regression operates on the assumption that certain variables can link to a discrete category.
It takes these variables, and uses them to predict the probability that the object belongs to a certain category. This probability is then used to categorise the object.

# The Sigmoid Function

## Equation

To predict the probability of an object being in a category, the sigmoid function is used:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

with z such that:

$$z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

Note:
$$0 < \sigma(z) < 1$$

The sigma function may be applied to probabilities due to its bounds.

## Explanation

In this formula, z is the log-odds of the positive class. It is the combined sum of the bias and the weighted variables.

$\beta_0$: Bias

$\beta_1, \dots, \beta_n$: Weights

$x_1, \dots, x_n$: Input features

The bias shifts the curve left or right along the axis to change the baseline probability. This allows the model to lean towards certain outcomes by default.

The weights allows the function to rely on certain input features more than others. This means that features that have a greater effect on the outcome impact the calculated probability more.

The input features are the variables used to calculate the probability. These variables must be independent to an extent to allow for accurate weight calculation. It is possible to model input features as polynomials or interactions depending on how they effect the outcome.

For example, if we use age to predict healthcare, we can model that as a polynomial with degree 2. Healthcare peaks early in life and late in life, while beint at a minimum during the middle of life. As this is not a linear relation we can model it as a polynomial to include it in the log-odds.

## Calculating Weights

### Maximum Likelihood Estimation

The Sigmoid function uses Maximum Likelihood to optimise the weights to calculate an accurate probability.

This process consists of two parts: the loss function, gradient descent.

### Log-loss Function

$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

$y$: True binary result

$\hat{y}$: Predicted probability

When the binary result is true:

$$
\begin{aligned}
y = 1 &\implies L = -\log(\hat{y}) \\
\text{as } \hat{y} \to 1 &\implies -\log(\hat{y}) \to 0 \\
\text{as } \hat{y} \to 0 &\implies -\log(\hat{y}) \to \infty
\end{aligned}
$$

When the binary result is false:

$$
\begin{aligned}
y = 0 &\implies L = -\log(1 - \hat{y}) \\
\text{as } \hat{y} \to 0 &\implies -\log(1 - \hat{y}) \to 0 \\
\text{as } \hat{y} \to 1 &\implies -\log(1 - \hat{y}) \to \infty
\end{aligned}
$$

This allows probabilities farther from the true result to receive greater punishment. '

To evaluate the accuracy of a set of weights, we can take the average of the Log-loss function for every set of input features in the test set. The smaller the Log-loss the more optimised the set of weights is. We can use this to check the gradient descent is working and to check whether a final set of weights has been reached.

The equation for the average of the Log-loss function:

$$J(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

This is useful for the gradient descent.

### Gradient Descent

As a generic summary, gradient descent essentially follows the gradient of a curve to find the minimum or maximum point. We can apply this to a Log-loss function to find its minimum point:

$$\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}$$

$\beta_j$: Optimised set of weights

$\alpha$: Learning rate

$\frac{\partial J}{\partial \beta_j}$: Deriviate of $J(\boldsymbol{\beta})$

$$\frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}$$

Note: We will not prove this, as it is beyond the aims of the research. $x_0 = 1$.

By plugging in our values, we can repeatedly use gradient descent to get closer to the optimised set of weights.

There is importance in how we choose $\alpha$. If it is too large, we risk large oscillations that never settle. If it is too small, then training will be extremely slow.

The ways to decide on an $\alpha$ value include fixed rate (choosing an appropriate number), adaptive optimisation (where $\alpha$ changes based on previous gradients and results) and logarithmic sampling (where different orders of magnitude are tested and the most appropriate order chosen).

This allows us to get closer to a Log-loss minimum and to find the optimal set of weights to use in our probability calculation.

Note: in order to increase the speed of calculations, weights and input features are often stored as matrices to allow for matrix calculations.

# .fit(X, y)

In pre-built models, such as scikit-learn, the models use a .fit() function to allow a model to learn. For logistic regression, this involves running gradient descent, using the formulas menntioned, to find optimal weights, minimising the Log-loss. After training, these weights are stored and then used to make predictions on new unseen data.

Logistic regression produces a probability representing the likelihood that an observation belongs to the positive class. To convert this probability into a binary prediction, an acceptance threshold is used. If the probability is above the threshold, the predicted class is 1. If it is below, the predicted class is 0. The threshold can be adjusted depending on the requirements of the model.

# Evaluation and the Confusion Matrix

In order to evaluate how well a logistic regression model performs, a Confusion Matrix may be used. This works by testing the model and recording the True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN). Calculations may be performed on this data to gain insight into how well the model has performed:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

# Regulisation

Some problems may arise when using logistic regression. Two of the main problems involve unimportant terms and unstable weights. The ways to counteracting these are L1 and L2.

L1: this involves reducing the less important terms down to 0 so that they do not effect the probability calculation.

L2: this involves reducing the coefficients of all variables so that they are more stable and balanced.

It is possible to combine the two by using an elastic net. An elastic net applies both L1 and L2 to the Log-loss function to help keep the weights stable, balanced and relevant.
