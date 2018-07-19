The entry point  is `src/sampling/apply_zv`. This algorithm takes one command line argument which is a number from 1-7 which simply specifies the stepsize to use.

Before the algorithm can be run the corresponding SGD optimiser needs to be run, which can be done by running the script `src/sgd_tuning/sgld.jl`. This again takes a number from 1-15 as a command line argument which specifies the stepsize and the dataset size.

The code expects a training and test set to be put in the data/ directory. This expects a sparse matrix of review data, the original dataset used was the [movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/).
