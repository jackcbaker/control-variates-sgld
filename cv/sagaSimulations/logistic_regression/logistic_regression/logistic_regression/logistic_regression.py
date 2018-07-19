import numpy as np
from stopwatch import Stopwatch
from saga import SAGA
from sklearn.metrics import log_loss


class LogisticRegression:
    """
    Methods for performing Bayesian logistic regression for large datasets.

    Logistic regression is trained using stochastic gradient Langevin dynamics
    with control variate postprocessing.

    References: 
        1. Logistic regression - https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
        2. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
        3. Control variates for MCMC -
                https://projecteuclid.org/download/pdfview_1/euclid.ba/1393251772
    """


    def __init__(self,X_train,X_test,y_train,y_test):
        """
        Initialise the logistic regression object.

        Parameters:
        X_train - matrix of explanatory variables for training (assumes numpy array of floats)
        X_test - matrix of explanatory variables for testing (assumes numpy array of ints)
        y_train - vector of response variables for training (assumes numpy array of ints)
        y_train - vector of response variables for testing (assumes numpy array of ints)
        """
        # Set error to be raised if there's an over/under flow
        np.seterr( over = 'raise', under = 'raise' )
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set dimension constants
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.test_size = self.X_test.shape[0]
        
        # Initialise containers
        # Logistic regression parameters (assume bias term encoded in design matrix)
        self.beta = np.zeros(self.d)
        # Storage for beta samples during fitting
        self.sample = None
        # Storage for logloss values during fitting
        self.training_loss = []
        self.n_iters = None
        self.fitter = None


    def fit(self,stepsize,n_iters=10**4,minibatch_size=500):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        # Holds log loss values once fitted
        self.training_loss = []
        # Number of iterations before the logloss is stored
        self.loss_thinning = 10
        # Initialize sample storage
        self.n_iters = n_iters
        self.sample = np.zeros( ( self.n_iters, self.d ) )
        self.grad_sample = np.zeros( ( self.n_iters, self.d ) )

        self.fitter = SAGA(self,stepsize,minibatch_size,n_iters)
        # Burn in chain
        print "Fitting chain..."
        print "{0}\t{1}".format( "iteration", "Test log loss" )
        timer = Stopwatch()
        for self.fitter.iter in range(1,n_iters+1):
            # Every so often output log loss on test set and store 
            if self.fitter.iter % self.loss_thinning == 0:
                elapsed_time = timer.toc()
                current_loss = self.logloss()
                self.training_loss.append( [current_loss,elapsed_time] )
                print "{0}\t\t{1}\t\t{2}".format( self.fitter.iter, current_loss, elapsed_time )
                timer.tic()
            self.fitter.update(self)
            self.sample[(self.fitter.iter-1),:] = self.beta


    def logloss(self):
        """Calculate the log loss on the test set, used to check convergence"""
        y_pred = np.zeros(self.test_size, dtype = int)
        for i in range(self.test_size):
            x = np.squeeze( np.copy( self.X_test[i,:] ) )
            y_pred[i] = int( np.dot( self.beta, x ) >= 0.0 )
        return log_loss( self.y_test, y_pred )


    def loglossp(self,beta):
        """
        Calculate the log loss on the test set for specified parameter values beta
        
        Parameters:
        beta - a vector of logistic regression parameters (float array)
        """
        y_pred = np.zeros(self.test_size, dtype = int)
        for i in range(self.test_size):
            x = np.squeeze( np.copy( self.X_test[i,:] ) )
            y_pred[i] = int( np.dot( beta, x ) >= 0.0 )
        return log_loss( self.y_test, y_pred )


    def dlogdens(self,sgld,indices = None):
        """
        Calculate gradient of the log density wrt the parameters at observations specified by indices

        Parameters:
        sgld - a StochasticGradientLangevinDynamics object, used to specify the minibatch

        Returns:
        dlogbeta - gradient of the log likelihood wrt the parameter beta at observation at each index
        """
        if not( indices ):
            indices = sgld.minibatch
        sample_size = len( indices )
        dlogbeta = np.zeros( ( sample_size, self.d ) )

        # Calculate sum of gradients at each point in the minibatch
        for i, index in enumerate(sgld.minibatch):
            x = np.squeeze( np.copy( self.X[index,:] ) )
            y = self.y[index]
            # Calculate gradient of the log density at current point, use to update dlogbeta
            # Handle overflow gracefully by catching numpy's error
            # (seterr was defined at start of class)
            dlogbeta[i,:] = ( y - 1 / ( 1 + np.exp( - np.dot( self.beta, x ) ) ) ) * x
        return dlogbeta
