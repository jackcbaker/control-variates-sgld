include("../simulate.jl")

function tune( model::matrix_factorisation, step )
    println( "Tuning sgld for $(model.name)" )
    out_dir = "../../../data/tuning/$(model.name)/cv/"
    mkpath(out_dir)
    println("Stepsize: $step")
    model = lda( model.train, model.test, model.K )
    out = sim_sgld( model, step, 50, 10^4 )
    writedlm("$out_dir/$step",out)
end

function test_run( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, seed::Int64 )
    println( "Running matrix factorization with $(model.L) users" )
    sgld_final( model, stepsize, sgd_step, 5000, 2*10^4, seed )
end

function truncate_data( n_users::Int64, train::Array{Int64,2}, test::Array{Int64,2} )
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    return( train, test )
end

function truncate_data( n_users::Int64, n_items::Int64, train::Array{Int64,2}, test::Array{Int64,2} )
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_item_in = train[:,2] .<= n_items
    train = train[is_item_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    is_item_in = test[:,2] .<= n_items
    test = test[is_item_in,:]
    @assert length( unique( train[:,1] ) ) == n_users
    return( train, test )
end
