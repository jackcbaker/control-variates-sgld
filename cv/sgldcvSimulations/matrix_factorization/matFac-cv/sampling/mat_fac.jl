using Iterators
include("tuning.jl")

index = parse( Int64, ARGS[1] )
stepsize_list = Dict( 10^2 => 5e-7,
                      5*10^2 => 1e-6,
                      NaN =>  1e-6 )
sgd_step_list = Dict( 10^2 => 1e-2,
                      5*10^2 => 1e-2,
                      NaN => 1e-2 )
seed_list = collect(1:5)
list_size = length( seed_list )
n_user_list = [10^2 5*10^2 NaN]
n_users = n_user_list[floor( Int64, ( index - 1 ) / list_size + 1 )]
sgd_step = sgd_step_list[n_users]
stepsize = stepsize_list[n_users]
seed_current = seed_list[floor( Int64, ( index - 1 ) % list_size + 1 )]
println( "Fitting with stepsize: $stepsize" )
train = readdlm( "../../data/datasets/ml-100k/u1.base" )
test = readdlm( "../../data/datasets/ml-100k/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
if !isnan(n_users)
    n_users = round( Int64, n_users )
    ( train, test ) = truncate_data( n_users, train, test )
end
srand(seed_current)
model = matrix_factorisation(train, test)
test_run(model,stepsize,sgd_step,seed_current)
