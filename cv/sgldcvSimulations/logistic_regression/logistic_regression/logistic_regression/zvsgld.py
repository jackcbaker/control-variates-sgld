import numpy as np
import sys
import pkg_resources
from stopwatch import Stopwatch
from sklearn.metrics import log_loss


class ZVSGLD:
    """
    Methods to apply SGLD with zero variance control variate postprocessing for logistic regression

    SGLD stands for stochastic gradient Langevin dynamics and is a MCMC method for large datasets.
    Zero variance control variates are used to improve the efficiency of the sample.

    SGLD notation used as in reference 1
    Zero variance control variate notation used as in reference 2
    References:
        1. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
        2. Zero variance control variates for Hamiltonian Monte Carlo - 
                https://projecteuclid.org/download/pdfview_1/euclid.ba/1393251772
    """
    
    def __init__(self,lr,epsilon,minibatch_size,n_iter):
        """
        Initialize the container for SGLD

        Parameters:
        lr - LogisticRegression object
        epsilon - the stepsize to perform SGD at
        minibatch_size - size of the minibatch used at each iteration
        n_iter - the number of iterations to perform
        """
        self.epsilon = epsilon
        # Set the minibatch size
        self.minibatch_size = minibatch_size
        self.sample_minibatch(lr)
        # Hold number of iterations so far
        self.iter = 1
        self.output = np.zeros( ( n_iter, lr.d ) )


    def update(self,lr):
        """
        Update one step of stochastic gradient Langevin dynamics

        Parameters:
        lr - LogisticRegression object

        Modifies:
        lr.beta - updates parameter values using SGLD
        lr.grad_sample - adds calculated gradient to storage
        """
        self.sample_minibatch(lr)
        # Calculate gradients at current point
        dlogbeta = lr.dlogpost(self)
        lr.grad_sample[self.iter-1,:] = dlogbeta

        # Update parameters using SGD
        eta = np.sqrt( self.epsilon ) * np.random.normal( size = lr.d )
        lr.beta += self.epsilon / 2 * dlogbeta + eta


    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
