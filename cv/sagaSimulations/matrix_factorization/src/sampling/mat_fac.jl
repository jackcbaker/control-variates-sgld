using Iterators
include("tuning.jl")

index = parse( Int64, ARGS[1] )
stepsize_list = Dict( 10^2 => 0.005,
                      5*10^2 => 0.01,
                      NaN =>  0.01 )
seed_list = collect(1:5)
list_size = length( seed_list )
seed_current = ( index - 1 ) % list_size + 1
n_user_list = [10^2 5*10^2 NaN]
n_users = n_user_list[floor( Int64, ( index - 1 ) / list_size + 1 )]
stepsize = stepsize_list[n_users]
train = readdlm( "../../data/datasets/ml-100k/u1.base" )
test = readdlm( "../../data/datasets/ml-100k/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
if !isnan(n_users)
    n_users = round( Int64, n_users )
    ( train, test ) = truncate_data( n_users, train, test )
end
model = MatFac(train, test)
println("Number of users: $(model.L)\tNumber of items: $(model.M)")
test_run(model,stepsize,seed_current)
