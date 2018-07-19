using JLD
include("../simulate.jl")

function tune( model::matrix_factorisation, step )
    println( "Tuning sgld for $(model.name)" )
    out_dir = "../../../data/tuning/$(model.name)/cv/"
    mkpath(out_dir)
    println("Stepsize: $step")
    model = lda( model.train, model.test, model.K )
    out = run_sgd( model, step, 50, 10^4, step )
    writedlm("$out_dir/$step",out)
end

function test_run( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64 )
    println( "Running matrix factorization with $(model.L) users" )
    sgd_init = run_sgd( model, stepsize, 5000, 3000 )
    save( "./sgd_log/$(model.L)/sgd_init-$stepsize.jld", "sgd_init", sgd_init )
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
