The entry point for the SGLD and SGLD-CV algorithms is `lda/sampling/lda.jl` and `cv-lda/sampling/lda.jl`. These algorithms take one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). 

Before SGLD-CV can be run the corresponding SGD optimiser needs to be run, which can be done by running `cv-lda/sgd_tuning/sgd.jl`. This again takes a number from 1-15 as a command line argument which specifies the stepsize and the dataset size.

The code expects a train and test dataset in the form of a document term matrix. Originally this was built by scraping Wikipedia. Please email `j` dot `baker` at `lancaster` dot `ac` dot `uk` if you require this dataset.
