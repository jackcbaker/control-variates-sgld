import numpy as np
import sys
import pkg_resources
from sklearn.metrics import log_loss


class SAGA:
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
        # Hold gradients of each data point
        self.g_alpha_i = lr.dlogdens(self,xrange(lr.N)) 
        self.g_alpha = self.g_alpha_i.sum(axis=0)


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
        # Calculate gradients of log density at current point and minibatch
        dlogdensgrads_beta = lr.dlogdens(self)
        # Calculate old and new log likelihood gradient estimates
        loglikgradest_beta = dlogdensgrads_beta.sum(axis=0)
        loglikgradest_alpha = self.g_alpha_i[self.minibatch,:].sum(axis=0)
        # Calculate SAGA estimate of log posterior gradient
        dlogbeta = self.dlogpostest(lr,loglikgradest_alpha,loglikgradest_beta)

        # Update g_alpha
        self.g_alpha += loglikgradest_beta - loglikgradest_alpha
        self.g_alpha_i[self.minibatch,:] = dlogdensgrads_beta

        # Update parameters using SGLD
        eta = np.random.normal( size = lr.d, scale = self.epsilon )
        lr.beta += self.epsilon / 2 * dlogbeta + eta


    def dlogpostest(self,lr,loglikgrad_alpha,loglikgrad_beta):
        """
        Calculate SAGA minibatch estimate of gradient of the log posterior wrt the parameters

        Parameters:
        sgld - a StochasticGradientLangevinDynamics object, used to specify the minibatch

        Returns:
        dlogbeta - estimated log posterior gradient
        """
        correction = lr.N / float( self.minibatch_size )
        dlogpostest_saga = self.g_alpha + correction *( loglikgrad_beta - loglikgrad_alpha )
        # Add gradient of log prior (assume Laplace prior with scale 1)
        dlogpostest_saga -= np.sign(lr.beta)
        return dlogpostest_saga


    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
