The entry point for the SGLD and SGLD-CV algorithms is `src/sampling/mat_fac.jl`. These algorithms take one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). 

The code expects a training and test set to be put in the data/ directory. This expects a sparse matrix of review data, the original dataset used was the [movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/).
