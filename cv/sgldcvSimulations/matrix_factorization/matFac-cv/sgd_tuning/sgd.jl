using Iterators
include("tuning.jl")

index = parse( Int64, ARGS[1] )
stepsize_list = Dict( 10^2 => [3e-3 5e-3 7e-3 1e-2 3e-2],
                      5*10^2 => [3e-3 5e-3 7e-3 1e-2 3e-2],
                      NaN =>  [3e-3 5e-3 7e-3 1e-2 3e-2] )
list_size = length( stepsize_list[NaN] )
n_user_list = [10^2 5*10^2 NaN]
list_size = length(stepsize_list[NaN])
n_users = n_user_list[floor( Int64, ( index - 1 ) / list_size + 1 )]
stepsize = stepsize_list[n_users][ ( index - 1 ) % list_size + 1]
sgd_step = NaN
train = readdlm( "../../data/datasets/ml-100k/u1.base" )
test = readdlm( "../../data/datasets/ml-100k/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
if !isnan(n_users)
    n_users = round( Int64, n_users )
    ( train, test ) = truncate_data( n_users, train, test )
end
model = matrix_factorisation(train, test)
srand(4)
println( "Number of users: $(model.L)\tNumber of items: $(model.M)" )
test_run(model,stepsize,sgd_step)
