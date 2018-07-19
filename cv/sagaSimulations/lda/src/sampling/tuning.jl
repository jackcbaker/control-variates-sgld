include("../simulate.jl")

function tune( model::lda, step::Float64, seed::Int64 )
    println( "Tuning sgld for $(model.name)" )
    println("Stepsize: $step")
    model = lda( model.train, model.test, model.K )
    sim_saga( model, step, 50, 4*10^4, seed )
end

function test_run( model::lda, stepsize, seed_curr )
    println( "Test run for $(model.name)" )
    sim_saga( model, stepsize, 50, 10^4, seed_curr )
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
