The entry point for the SAGA algorithm is `logistic_regression/simulation/cover_type_saga.py`. This algorithm takes one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). The scripts can be run by issuing the command `python -m logistic_regression.simulation.cover_type_sgld`.

There is code in the script to automatically download the required covertype dataset.
