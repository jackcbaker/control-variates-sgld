include("../simulate.jl")

function tune( model::lda, step, sgdstep, seed )
    println( "Tuning sgld for $(model.name)" )
    println("Stepsize: $step")
    model = lda( model.train, model.test, model.K )
    sgld_sgd_init( model, step, sgdstep, 50, 2*10^4, seed )
end

function test_run( model::lda, stepsize )
    println( "Test run for $(model.name)" )
    sim_sgld( model, stepsize, 50, 10^3 )
end

function truncate( train::Array{Int64,2}, test::Array{Int64,2}, N::Float64 )
    return ( train, test )
end

function truncate( train::Array{Int64,2}, test::Array{Int64,2}, N::Int64 )
    sample_indexes = sample( 1:size( train, 1 ), N, replace = false )
    train = train[sample_indexes,:]
    test = test[sample_indexes,:]
    return( train, test )
end
