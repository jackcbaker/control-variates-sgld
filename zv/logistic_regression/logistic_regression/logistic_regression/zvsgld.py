import numpy as np
import sys
import pkg_resources
from sklearn.covariance import LedoitWolf
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
        dlogbeta, dlogbetaopt = lr.dlogpostcv(self)
        lr.grad_sample[self.iter-1,:] = dlogbeta

        # Update parameters using SGD
        eta = np.sqrt( self.epsilon ) * np.random.normal( size = lr.d )
        lr.beta += self.epsilon / 2 * ( lr.full_post + ( dlogbeta - dlogbetaopt ) ) + eta

    
    def full_post(self,lr):
        self.minibatch = np.arange(lr.N)
        dlogbeta, dlogbetaopt = lr.dlogpostcv(self)
        lr.full_post = self.minibatch_size / float( lr.N ) * dlogbetaopt

    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )


    def control_variates(self,lr):
        """
        Postprocess a fitted LogisticRegression object using zero variance control variates.

        Assumes object has already been fitted using SLGD i.e. lr.sample is nonempty.

        Parameters:
        lr - fitted LogisticRegression object

        Modifies:
        lr.sample - updates stored MCMC chain using ZV control variates
        """
        pot_energy = - 1 / 2.0 * lr.grad_sample
        sample_mean = np.mean( lr.sample, axis = 0 )
        grad_mean = np.mean( pot_energy, axis = 0 )
        var_grad_inv = self.shrinkage_precision(pot_energy)

        # Initialise variables
        cov_params = np.zeros( lr.d )
        a_current = np.zeros( lr.d )
        new_sample = np.zeros( lr.sample.shape )

        # Calculate covariance for each parameter
        print "Calculating control variates..."
        for j in range(lr.d):
            sys.stdout.write("{0} ".format(j))
            sys.stdout.flush()
            cov_params = np.zeros(lr.d)
            a_current = np.zeros(lr.d)
            for i in range(lr.n_iters):
                cov_params += 1 / float( lr.n_iters - 1 ) * ( 
                        lr.sample[i,j] - sample_mean[j] ) * ( pot_energy[i,:] - grad_mean )
            # Update sample for current dimension
            a_current = - np.matmul( var_grad_inv, cov_params )
            for i in range(lr.n_iters):
                new_sample[i,j] = lr.sample[i,j] + np.dot( a_current, pot_energy[i,:] )
        print
        # Compare new samples
        sample_size = 20
        random_points = np.random.choice( range(lr.n_iters), sample_size )
        llold = np.zeros( sample_size )
        llnew = np.zeros( sample_size )
        print "Calculating new log loss values..."
        for i,index in enumerate(random_points):
            sys.stdout.write("{0} ".format(i))
            sys.stdout.flush()
            llold[i] = lr.loglossp( lr.sample[index,:] )
            llnew[i] = lr.loglossp( new_sample[index,:] )
        print
        return llold, llnew


    def shrinkage_precision(self,PE):
        """Calculate estimate of precision of PE using shrinkage"""
        model = LedoitWolf()
        model.fit(PE)
        return model.get_precision()
