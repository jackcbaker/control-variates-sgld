using Iterators
using JLD
include("tuning.jl")

index = parse( Int64, ARGS[1] )
stepsize_list = Dict( 10^4 => [1e-4 3e-4 5e-4 7e-4 1e-3],
                      5*10^4 => [3e-5 5e-5 8e-5 1e-4 3e-4],
                      NaN =>  [5e-5 8e-5 1e-4 3e-4 5e-4] )
n_user_list = [10^4 5*10^4 NaN]
size_list = length( stepsize_list[NaN] )
n_users = n_user_list[floor( Int64, ( index - 1 ) / size_list + 1 )]
stepsize = stepsize_list[n_users][ ( index - 1 ) % size_list + 1]
if !isnan( n_users )
    n_users = floor( Int64, n_users )
end
train = readdlm("../../data/datasets/train" )
test = readdlm("../../data/datasets/test" )
train = round( Int64, train )
test = round( Int64, test )
srand( 13 )
( train, test ) = truncate( train, test, n_users )
model = lda(train, test, 20)
sgd_init = run_sgd( model, stepsize, 50, 10^4 )
save( "./sgd_out/$(model.M)/sgd_init-$stepsize.jld", "sgd_init", sgd_init )
