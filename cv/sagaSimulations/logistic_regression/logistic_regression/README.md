The entry point for the SAGA algorithms is `logistic_regression/simulation/cover_type_sgld.py` and `logistic_regression_cv/simulation/cover_type_sgld_cv.py`. These algorithms take one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). The scripts can be run by issuing the command `python -m logistic_regression.simulation.cover_type_sgld`.

Before SGLD-CV can be run the corresponding SGD optimiser needs to be run, which can be done by running the script `logistic_regression_cv/simulation/cover_sgd.py`. This again takes a number from 1-15 as a command line argument which specifies the stepsize and the dataset size.

There is code in the script to automatically download the required covertype dataset.
