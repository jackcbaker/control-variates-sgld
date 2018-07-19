import os
import random
import sys
import pickle
import pkg_resources
import urllib
import numpy as np
import bz2
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from ..logistic_regression.logistic_regression import LogisticRegression


class CoverType:
    """
    Example for fitting a logistic regression model to the cover type dataset

    References:
    1. Cover type dataset - https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    """

    def __init__(self):
        """Load data into the object"""
        self.data_dir = pkg_resources.resource_filename('logistic_regression', 'data/')
        self.lr = None
        # Try opening the data, if it's not available, download it.
        try:
            np.load( self.data_dir + 'cover_type/X_train.dat' )
        except IOError:
            raw = self.download_data()
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()
            self.X_train.dump( self.data_dir + 'cover_type/X_train.dat' )
            self.X_test.dump( self.data_dir + 'cover_type/X_test.dat' )
            self.y_train.dump( self.data_dir + 'cover_type/y_train.dat' )
            self.y_test.dump( self.data_dir + 'cover_type/y_test.dat' )
        else:
            self.X_train = np.load( self.data_dir + 'cover_type/X_train.dat' )
            self.X_test = np.load( self.data_dir + 'cover_type/X_test.dat' )
            self.y_train = np.load( self.data_dir + 'cover_type/y_train.dat' )
            self.y_test = np.load( self.data_dir + 'cover_type/y_test.dat' )
    
    
    def truncate(self,train_size,test_size):
        self.X_train = self.X_train[:train_size,:]
        self.y_train = self.y_train[:train_size]
        self.X_test = self.X_test[:test_size,:]
        self.y_test = self.y_test[:test_size]


    def fit(self,stepsize):
        """
        Fit a Bayesian logistic regression model to the data using the LogisticRegression class.

        Parameters:
        stepsize - stepsize parameter for the stochastic gradient langevin dynamics

        Returns:
        lr - fitted LogisticRegression object
        """
        self.lr = LogisticRegression( self.X_train, self.X_test, self.y_train, self.y_test )
        self.lr.fit(stepsize, n_iters = 2*10**4)


    def download_data(self):
        """Download raw cover type data"""
        if not os.path.exists( self.data_dir + 'cover_type' ):
            os.makedirs( self.data_dir + 'cover_type' )
        print "Downloading data..."
        urllib.urlretrieve( ( "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
                "datasets/binary/covtype.libsvm.binary.scale.bz2" ), 
                self.data_dir + "cover_type/covtype.libsvm.binary.bz2" )
        compressed_file = bz2.BZ2File( self.data_dir + 'cover_type/covtype.libsvm.binary.bz2', 
                'r')
        with open( self.data_dir + 'cover_type/raw.dat', 'w' ) as out: 
            out.write( compressed_file.read() )


    def preprocess(self):
        """Preprocess raw data once downloaded, split into train and test sets"""
        X, y = load_svmlight_file( self.data_dir + 'cover_type/raw.dat' )
        X_train, X_test, y_train, y_test = train_test_split( X, y  )
        # Set y to go from 0 to 1
        y_train -= 1
        y_test -= 1
        # Add a bias term to X
        X_train = np.concatenate( ( np.ones( ( len(y_train), 1 ) ), X_train.todense() ), axis = 1 )
        X_test = np.concatenate( ( np.ones( ( len(y_test), 1 ) ), X_test.todense() ), axis = 1 )
        return X_train, X_test, y_train.astype(int), y_test.astype(int)


    def simulation(self,index):
        print "Simulating..."
        outdir = 'cover_type_sgld'
        n_obs_list = [0.01, 0.1, 1]
        step_list = { 0.01: 0.0005, 0.1: 5e-05, 1: 5e-06 }
        seed_list = range(1,6)
        n_seeds = len( seed_list )
        n_obs = n_obs_list[index / n_seeds]
        stepsize = step_list[n_obs]
        seed_current = seed_list[index % n_seeds]
        train_size = int( n_obs * self.X_train.shape[0] )
        test_size = int( n_obs * self.X_test.shape[0] )
        self.truncate( train_size, test_size )
        random.seed(seed_current)
        self.fit(stepsize)
        if not os.path.exists( self.data_dir + outdir + '/{0}/'.format(n_obs) ):
            os.makedirs( self.data_dir + outdir + '/{0}/'.format(n_obs) )
        np.savetxt( self.data_dir + outdir + '/{0}/{1}.dat'.format(n_obs,seed_current), np.array( self.lr.training_loss ) )


if __name__ == '__main__':
    print "Simulation started!"
    index = int( sys.argv[1] ) - 1
    example = CoverType()
    example.simulation(index)
