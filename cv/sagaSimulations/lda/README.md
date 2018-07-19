The entry point for the SAGA algorithm is `src/sampling/lda.jl`. These algorithms take one command line argument which is a number from 1-15 which simply specifies the seed value and dataset size for the run (3 dataset sizes and 5 seeds). 

The code expects a train and test dataset in the form of a bag of words matrix. Originally this was built by scraping Wikipedia. Please email `j` dot `baker` at `lancaster` dot `ac` dot `uk` if you require this dataset.
