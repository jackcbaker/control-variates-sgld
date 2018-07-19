The entry point for the SGLD and SGLD-CV algorithms is `matFac/sampling/mat_fac.jl` and `matFac-cv/sampling/mat_fac.jl`. These algorithms take one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). 

Before SGLD-CV can be run the corresponding SGD optimiser needs to be run, which can be done by running `matFac-cv/sgd_tuning/sgd.jl`. This again takes a number from 1-15 as a command line argument which specifies the stepsize and the dataset size.

The code expects a training and test set to be put in the data/ directory. This expects a sparse matrix of review data, the original dataset used was the [movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/).
