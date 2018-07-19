using JLD
include("../simulate.jl")

job_id = parse( Int64, ARGS[1] )
sgd_step = 3e-4
stepsize_list = [1e-6 3e-6 5e-6 7e-6 1e-5 3e-5 5e-5]
list_size = length(stepsize_list)
seed_curr = floor( Int64, job_id / list_size ) + 1
stepsize = stepsize_list[(job_id - 1) % list_size + 1]
println( "Fitting with stepsize: $stepsize" )
train = readdlm( "../../data/datasets/ml-100k/u1.base" )
test = readdlm( "../../data/datasets/ml-100k/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
srand( seed_curr )
model = matrix_factorisation(train, test)
( rmse_old, rmse_new ) = sgld_sgd_init( model, stepsize, sgd_step, 5000, 10^4 )
mkpath("zv/$stepsize/")
save( "zv/$stepsize/rmse-$seed_curr.jld", "old", rmse_old, "new", rmse_new )
