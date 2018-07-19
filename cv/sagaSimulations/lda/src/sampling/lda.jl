using Iterators
include("tuning.jl")

index = parse( Int64, ARGS[1] )
stepsize_list = Dict( 10^4 => 5e-4,
                      5*10^4 => 5e-5,
                      NaN =>  5e-5 )
seed_list = collect(1:5)
n_user_list = [10^4 5*10^4 NaN]
size_list = length( seed_list )
n_users = n_user_list[floor( Int64, ( index - 1 ) / size_list + 1 )]
seed = ( index - 1 ) % size_list + 1
stepsize = stepsize_list[n_users]
if !isnan( n_users )
    n_users = floor( Int64, n_users )
end
train = readdlm("../../data/datasets/train" )
test = readdlm("../../data/datasets/test" )
train = round( Int64, train )
test = round( Int64, test )
( train, test ) = truncate( train, test, n_users )
srand(seed) 
model = lda(train, test, 20)
tune(model,stepsize,seed)
