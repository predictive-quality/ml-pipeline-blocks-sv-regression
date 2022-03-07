# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from absl import app, flags, logging
from sv_regression import Regression
from s3_smart_open import to_txt, read_pd_fth, to_json, to_pd_fth
import pandas as pd

flags.DEFINE_string('model_name',None,'Name of the model to save/load')
flags.DEFINE_string('input_path',None,'Specify the a local or s3 object storage path where the input files are stored')
flags.DEFINE_string('output_path',None,'Specify the path where the output files will be stored')
flags.DEFINE_string('filename_x',None,'Filename of Dataframe with feautres')
flags.DEFINE_string('filename_y',None,'Filename of Dataframe with the target')
flags.DEFINE_string('y_col_name',None,'Name of the y column to train or to predict')
flags.DEFINE_enum('stage',None,['fit','predict','evaluate'],'Wether to fit, predict or evaluate the regressor')
flags.DEFINE_enum('method',None,['SVR','linearSVR'],'Wether to use sklearn.svm.linearSVR() or sklearn.svm.SVR()')
flags.DEFINE_float('tol',1e-3,'Tolerance for stopping criterion.')
flags.DEFINE_float('C',1.0,'Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.')
flags.DEFINE_integer('verbose',0,'Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm/liblinear that, if enabled, may not work properly in a multithreaded context.')
flags.DEFINE_integer('max_iter',1000,'Hard limit on iterations within solver, or (only method epsilon) -1  for no limit.')
flags.DEFINE_enum('kernel','rbf',['linear','poly','rbf','sigmoid','precomputed'],'Method epsilon: Specifies the kernel type to be used in the algorithm.')
flags.DEFINE_integer('degree',3,'Method epsilon: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')
flags.DEFINE_string('gamma','scale','Method epsilon: Kernel coefficient for rbf, poly and sigmoid.')
flags.DEFINE_float('coef0',0.0,'Method epsilon: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.')
flags.DEFINE_boolean('shrinking',True,'Method epsilon: Whether to use the shrinking heuristic')
flags.DEFINE_float('cache_size',200,'Method epsilon: Specify the size of the kernel cache (in MB).')
flags.DEFINE_float('epsilon',0.0,'Epsilon parameter in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y. If unsure, set epsilon=0.')
flags.DEFINE_enum('loss','epsilon_insensitive',['epsilon_insensitive','squared_epsilon_insensitive'],'Method linear:Specifies the loss function. The epsilon-insensitive loss (standard SVR) is the L1 loss, while the squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss.')
flags.DEFINE_boolean('fit_intercept',True,'Method linear: Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).')
flags.DEFINE_float('intercept_scaling',1.0,'Method linear: To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.')
flags.DEFINE_boolean('dual',False,'Method linear: Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.')
flags.DEFINE_integer('random_state',None,'Method linear: Controls the pseudo random number generation for shuffling the data. Pass an int for reproducible output across multiple function calls.')
flags.DEFINE_list('metrics',None,'Metrics for on training evaluation. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/metrics')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('input_path')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('stage')
flags.mark_flag_as_required('filename_x')


def main(argv):
    """Fit, predict or evaluate datasets with a sklearn Support Vector Regression object.
    """    
    del argv

    to_txt(FLAGS.output_path,'flags_'+FLAGS.stage+'.txt',FLAGS.flags_into_string())
    X = read_pd_fth(FLAGS.input_path, FLAGS.filename_x)

    if FLAGS.stage == 'fit':
        if FLAGS.method == 'SVR' and X.shape[0] > 20000:
            logging.warning('Consider to use Linear Support Vector Regression for datasets with more than 20000 Samples. Change method to linear.')    
            
        regr = Regression.factoryRegression(model_name=FLAGS.model_name,
                                            method=FLAGS.method,
                                            tolerance=FLAGS.tol,
                                            C=FLAGS.C,
                                            verbose=FLAGS.verbose,
                                            max_iter=FLAGS.max_iter,
                                            kernel=FLAGS.kernel,
                                            degree=FLAGS.degree,
                                            gamma=FLAGS.gamma,
                                            coef0=FLAGS.coef0,
                                            shrinking=FLAGS.shrinking,
                                            cache_size=FLAGS.cache_size,
                                            epsilon=FLAGS.epsilon,
                                            loss=FLAGS.loss,
                                            fit_intercept=FLAGS.fit_intercept,
                                            intercept_scaling=FLAGS.intercept_scaling,
                                            dual=FLAGS.dual,
                                            random_state=FLAGS.random_state)

        y = read_pd_fth(FLAGS.input_path, FLAGS.filename_y, FLAGS.y_col_name, col_limit=1)
        regr.fit(X,y)
        regr.save(FLAGS.output_path)
        
        return

    regr = Regression.load(FLAGS.input_path,FLAGS.model_name)

    if FLAGS.stage == 'predict':
        y_pred = regr.predict(X)
        df_pred = pd.DataFrame(y_pred,columns=['Prediction'])
        to_pd_fth(FLAGS.output_path,FLAGS.model_name+'_results.fth',df_pred)
        logging.info(df_pred.head(n=5))

    if FLAGS.stage == 'evaluate':
        y = read_pd_fth(FLAGS.input_path, FLAGS.filename_y, FLAGS.y_col_name, col_limit=1)
        res = regr.evaluate(X,y,FLAGS.metrics)
        to_json(FLAGS.output_path,FLAGS.model_name+'_metrics.json',res)
        logging.info('Metrics: {}'.format(res))

if __name__ == '__main__':
    app.run(main)