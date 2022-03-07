# Support Vector Regression
## Epsilon-Support Vector Regression

The free parameters in the model are C and epsilon.

The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to datasets with more than a couple of 10000 samples. For large datasets consider using LinearSVR or SGDRegressor instead, possibly after a Nystroem transformer.

Read more in the [User Guide](https://scikit-learn.org/stable/modules/svm.html#svm-regression).
## Linear Support Vector Regression

Similar to SVR with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

This class supports both dense and sparse input.

Read more in the [User Guide](https://scikit-learn.org/stable/modules/svm.html#svm-regression).

Note: sklearn.svm.LinearSVR().fit() and sklearn.svm.SVR().fit() are implemented without fit parameter `sample_weight` at the moment.    
## Installation

Clone the repository and install all requirements using `pip install -r requirements.txt` .


## Usage

As mentioned above it is possible to to run the epsilon support vector regression with linear kernal to get a linear support vector model.
Due to bad scaling when the dataset has many samples it is reommended to use method `linear` instead of method `epsilon` with kernel `linear`!

You can run the code in two ways.
1. Use command line flags as arguments `python main.py --input_path= --output_path=...`
2. Use a flagfile.txt which includes the arguments `python main.py --flagfile=example/flagfile.txt`

## Input Flags/Arguments

#### --model_name
Name of the model to save/load.

#### --input_path
Specify the a local or s3 object storage path where the input files are stored.
For a s3 object storage path a valid s3 configuration is required.

#### --output_path
Specify the path where the output files will be stored.
For a s3 object storage path a valid s3 configuration is required.

#### --filename_x
Filename of Dataframe with feautres.

#### --filename_y
Filename of Dataframe with the target.

#### --y_col_name
Name of the y column to train or to predict.
#### --stage
Wether to fit, predict or evaluate the model
#### --method
Wether to use sklearn.svm.linearSVR() or sklearn.svm.SVR()
#### --tol
Tolerance for stopping criterion.

#### --verbose
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm/liblinear that, if enabled, may not work properly in a multithreaded context.

#### --metrics
Metrics for model evaluation. See [here](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) for candidate functions.

### Epsilon method
#### --C
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
#### --max_iter
Hard limit on iterations within solver, or -1 for no limit.
#### --kernel
Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.

#### --degree
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
#### --gamma
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

 - if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamme
 - if ‘auto’, uses 1 / n_features.
#### --coef0
Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
#### --shrinking
Whether to use the shrinking heuristic. See the [User Guide](https://scikit-learn.org/stable/modules/svm.html#svm-regression).
#### --cache_size
Specify the size of the kernel cache (in MB).
#### --epsilon
Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.

### Linear method
#### --C
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
#### --max_iter
The maximum number of iterations to be run.
#### --epsilon
Epsilon parameter in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y. If unsure, set epsilon=0.
#### --loss
Specifies the loss function. The epsilon-insensitive loss (standard SVR) is the L1 loss, while the squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss.
#### --fit_intercept
Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).
#### --intercept_scaling
When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
#### --dual
Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
#### --random_state
Controls the pseudo random number generation for shuffling the data. Pass an int for reproducible output across multiple function calls. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).



## Example

First move to the repository directory. \
Now you can fit an epsilon support vector model by using `python main.py --flagfile=example/ff_train_SVR.txt`. \
To fit a linear support vector model use `python main.py --flagfile=example/ff_train_linearSVR.txt`. \
After a model was fitted use `python main.py --flagfile=example/ff_evaluate_SVR.txt` to evaluate the fitted model \
or use `python main.py --flagfile=example/ff_predict_SVR.txt` to predict some targets.

## Data Set

The data set was recorded with the help of the Festo Polymer GmbH. The features (`x.csv`) are either parameters explicitly set on the injection molding machine or recorded sensor values. The target value (`y.csv`) is a crucial length measured on the parts. We measured with a high precision coordinate-measuring machine at the Laboratory for Machine Tools (WZL) at RWTH Aachen University.