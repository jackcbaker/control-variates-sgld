import os
import random
import sys
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


    def fit(self,stepsize,sgd_step):
        n_obs = self.X_train.shape[0]
        self.lr = LogisticRegression( self.X_train, self.X_test, self.y_train, self.y_test )
        beta_mode = np.load( "{0}cover_type_mode/{1}/{2}.npy".format(self.data_dir,n_obs,sgd_step) )
        self.lr.fit(stepsize,beta_mode,10**4)


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


    def simulation_step(self,index):
        print "Simulating..."
        outdir = 'cover_type_sgld_zv'
        sgd_step = 5e-6
        stepsize_list = [1e-6, 3e-6, 5e-6, 8e-6, 1e-5, 3e-5, 5e-5]
        n_stepsizes = len(stepsize_list)
        seed_current = index / n_stepsizes + 1
        stepsize = stepsize_list[index % n_stepsizes]
        print "Stepsize: {0}\tSeed: {1}".format(stepsize, seed_current)
        random.seed(seed_current)
        self.fit(stepsize,sgd_step)
        llold, llnew = self.lr.postprocess() 
        try:
            os.makedirs( self.data_dir + outdir + '/{0}/'.format(stepsize) )
        except OSError:
            pass
        np.savetxt( self.data_dir + outdir + '/{0}/old-{1}.dat'.format(stepsize, seed_current), np.array(llold))
        np.savetxt( self.data_dir + outdir + '/{0}/new-{1}.dat'.format(stepsize, seed_current), np.array(llnew))


if __name__ == '__main__':
    print "Simulation started!"
    index = int( sys.argv[1] ) - 1
    example = CoverType()
    example.simulation_step(index)
