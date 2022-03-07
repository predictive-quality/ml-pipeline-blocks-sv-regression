# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from sklearn.svm import LinearSVR, SVR
from s3_smart_open import to_pckl, read_pckl, read_pd_fth
from absl import logging
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow import keras

class Regression(BaseEstimator, RegressorMixin):
    """Regression class built with sklearn
    """   
    def __init__(self, model_name, tolerance, C, verbose, max_iter,  epsilon):
        """Initialize shared parameters for Support Vector Regression.
        Args:
            model_name (str): Name of the model to save/load.
            tolerance (float): Tolerance for stopping criterion.
            C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
            verbose (int): Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm/liblinear that, if enabled, may not work properly in a multithreaded context.
            max_iter (int): Hard limit on iterations within solver, or (only method epsilon) -1  for no limit.
            epsilon (float): Epsilon parameter in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y.
        """    
        self.tol = tolerance
        assert epsilon >= 0
        self.epsilon = epsilon
        assert C >= 0
        self.C = C
        self.max_iter = max_iter
        self.fitted_ = False
        self.model_name = model_name
        self.verbose = verbose


    def build_model(self):
        raise NotImplementedError


    @staticmethod
    def factoryRegression(model_name, method, tolerance, C, verbose, max_iter, kernel, degree, gamma, coef0, shrinking, cache_size, epsilon, loss, fit_intercept, intercept_scaling, dual, random_state):
        """Wehter to use sklearn.svm.SVR() or sklearn.svm.linearSVR() as model.
        Args:
            Please find descriptions for arguments in the subclasses 
        Returns:
            [Regression object]: Epsilon(SVR) or Linear SV Regression object to use for fit, predict and evaluate.
        """       
        if method == 'SVR':
            return SVRegression(model_name, tolerance , C, epsilon, verbose, max_iter, kernel, degree, gamma, coef0, shrinking, cache_size)
        elif method == 'linearSVR':
            return linearSVRegression(model_name, tolerance, C, epsilon, verbose, max_iter, loss, fit_intercept , intercept_scaling, dual, random_state)
        else:
            raise Exception('Method {} is invalid'.format(method))
        
    def fit(self,X,y):
        """Build and fit the model to a given dataset X (features) and y (targets).
        Args:
            X (pd.DataFrame): features data
            y (pd.DataFrame): target data
        """
        X = X.values
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        self.model = self.build_model()
        logging.info('Begin fitting with '+str(X.shape)+' samples')
        self.model.fit(X,y)
        self.fitted_ = True
    
    def predict(self,X):
        """Predict the targets given a set of inputs with already fitted model.

        Args:
            X (pd.DataFrame): Features

        Returns:
            [pd.DataFrame]: Predicted targets
        """    
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'fitted_')
        
        return self.model.predict(X)

    def evaluate(self,X,y,metrics):
        """Evaluate a fitted model and return given keras metrics.

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Targets
            metrics (list[str]): keras.metrics

        Returns:
            [dict]: Metrics in format {key,value}
        """
        X = X.values
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        check_is_fitted(self, 'fitted_')
        y_pred = self.model.predict(X)
        metrics = [keras.metrics.get(m) for m in metrics]
        res = {}
        for m in metrics:
            m.update_state(y, y_pred)
            res[m.name] = m.result().numpy().tolist()        
        return res

    def save(self,output_path):
        """Saves the Regression object to disk or to s3 bucket.

        Args:
            output_path (str): Path to storage location.
        """
        to_pckl(output_path,self.model_name+'.pckl',self)

    @staticmethod
    def load(input_path:str, model_name:str):
        """Loads the Regression object.

        Args:
            input_path (str): Path where the object is stored
            model_name (str): Name of the object to load

        Returns:
            [sv_regression.Regression object]: Regression object to use for predict and evaluate.
        """
        model = read_pckl(input_path, model_name+'.pckl')
        return model
    
    
class SVRegression(Regression):
    """Scikit learn Epsilon Support Vector Regression Model
    Args:
        Regression (class): Regression class built with sklearn
    """
    def __init__(self,model_name, tolerance, C, epsilon, verbose, max_iter, kernel, degree, gamma, coef0, shrinking, cache_size):
        """Initialize  Epsilon Support Vector Regression
        Args:
            model_name (str): Name of the model to save/load.
            tolerance (float): Tolerance for stopping criterion.
            C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
            verbose (int): Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm/liblinear that, if enabled, may not work properly in a multithreaded context.
            max_iter (int): Hard limit on iterations within solver, or (only method epsilon) -1  for no limit.
            kernel (string): Specifies the kernel type to be used in the algorithm.
            degree (int): Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            gamma (float or str): Kernel coefficient for rbf, poly and sigmoid.
            coef0 (float): Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            shrinking (bool): Whether to use the shrinking heuristic.
            cache_size (float): Specify the size of the kernel cache (in MB).
            epsilon (float): in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y.
        """            
        assert (max_iter > 0 or max_iter == -1)
        super().__init__(model_name, tolerance, C, verbose, max_iter, epsilon)
        self.kernel = kernel
        assert degree >= 0
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        assert cache_size > 0
        self.cache_size = cache_size
        self.verbose = bool(self.verbose)


    def build_model(self):
        """Builds model from sklearn.svm.SVR(), 
        """  
        if self.gamma not in ['scale','auto']:
            self.gamma = float(self.gamma)
            assert self.gamma > 0
        model = SVR(tol=self.tol,
                    C=self.C,
                    max_iter=self.max_iter,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    epsilon=self.epsilon,
                    shrinking=self.shrinking,
                    cache_size=self.cache_size,
                    verbose=self.verbose
                    )
        return model


class linearSVRegression(Regression):
    """Scikit learn linear Support Vector Regression Model
    Args:
        Regression (class): Regression class built with sklearn
    """
    def __init__(self, model_name, tolerance, C, epsilon, verbose, max_iter, loss, fit_intercept, intercept_scaling, dual, random_state):
        """Initialize Linear Support Vector Regression

        Args:
            model_name (str): Name of the model to save/load.
            method (str): Wether to use sklearn.svm.linearSVR() or sklearn.svm.SVR()
            tolerance (float): Tolerance for stopping criterion.
            C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
            verbose (int): Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm/liblinear that, if enabled, may not work properly in a multithreaded context.
            max_iter (int): Hard limit on iterations within solver, or (only method epsilon) -1  for no limit.
            epsilon (float): Epsilon parameter in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y.
            loss (str): Specifies the loss function. The epsilon-insensitive loss (standard SVR) is the L1 loss, while the squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss.
            fit_intercept (bool): Whether to calculate the intercept for this model.
            intercept_scaling (float): To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
            dual (bool): Select the algorithm to either solve the dual or primal optimization problem.
            random_state (int): Controls the pseudo random number generation for shuffling the data. Pass an int for reproducible output across multiple function calls.
        """    
        assert max_iter > 0, 'No Limit for linear support vector regression is not allowed!'
        super().__init__(model_name, tolerance, C, verbose, max_iter, epsilon)
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.random_state = random_state
        self.verbose = verbose


    def build_model(self):
        """Builds model from sklearn.svm.linearSVR(), 
        """    
        model = LinearSVR(tol=self.tol,
                            C=self.C,
                            max_iter=self.max_iter,
                            epsilon=self.epsilon,
                            loss=self.loss,
                            fit_intercept=self.fit_intercept,
                            intercept_scaling=self.intercept_scaling,
                            dual=self.dual,
                            random_state=self.random_state, 
                            verbose=self.verbose
                            )
        
        return model
