include("../simulate.jl")

function truncate( train::Array{Int64,2}, test::Array{Int64,2}, N::Float64 )
    return ( train, test )
end

function truncate( train::Array{Int64,2}, test::Array{Int64,2}, N::Int64 )
    sample_indexes = sample( 1:size( train, 1 ), N, replace = false )
    train = train[sample_indexes,:]
    test = test[sample_indexes,:]
    return( train, test )
end
