The entry point  is `logistic_regression/simulation/cover_type_sgld_zv.py`. This algorithm takes one command line argument which is a number from 1-7 which simply specifies the stepsize to use. The scripts can be run by issuing the command `python -m logistic_regression.simulation.cover_type_sgld_zv`.

Before the algorithm can be run the corresponding SGD optimiser needs to be run, which can be done by running the script `logistic_regression_cv/simulation/cover_sgd.py`. This again takes a number from 1-30 as a command line argument which specifies the stepsize and the dataset size.

There is code in the script to automatically download the required covertype dataset.
