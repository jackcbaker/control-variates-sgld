include("matrix_factorisation.jl")

"""
Container for sgld parameters
"""
type sgd
    subsize::Int64                  # Minibatch size
    iter::Int64
    subsample::Array{ Int64, 1 }
    ϵ_U::Float64             # Stepsize tuning constants
    G_U::Float64
    ϵ_V::Float64             # Stepsize tuning constants
    G_V::Float64
    ϵ_a::Float64             # Stepsize tuning constants
    G_a::Float64
    ϵ_b::Float64             # Stepsize tuning constants
    G_b::Float64
    ll_U::Array{ Float64, 2 }
    ll_V::Array{ Float64, 2 }
    ll_a::Array{ Float64, 1 }
    ll_b::Array{ Float64, 1 }
    U::Array{ Float64, 2 }
    V::Array{ Float64, 2 }
    a::Array{ Float64, 1 }
    b::Array{ Float64, 1 }
end

function sgd( model::matrix_factorisation, subsize::Int64, opt_stepsize::Float64 )
    subsample = sample( 1:model.N, subsize )
    ϵ_U = opt_stepsize
    ϵ_V = opt_stepsize
    ϵ_a = opt_stepsize
    ϵ_b = opt_stepsize
    G_U = 0
    G_V = 0
    G_a = 0
    G_b = 0
    ll_U = zeros( size(model.U) )
    ll_V = zeros( size(model.V) )
    ll_a = ones( size(model.a) )
    ll_b = ones( size(model.b) )
    U = zeros( size(model.U) )
    V = zeros( size(model.V) )
    a = zeros( size(model.a) )
    b = zeros( size(model.b) )
    sgd( subsize, 1, subsample, ϵ_U, G_U, ϵ_V, G_V, ϵ_a, G_a, ϵ_b, G_b, ll_U, ll_V, ll_a, ll_b,
            U, V, a, b )
end

function dlogpostΛ( model::matrix_factorisation )
    dlogΛ_U = zeros(model.D)
    dlogΛ_V = zeros(model.D)
    for d in 1:model.D
        dlogΛ_U[d] = ( model.α + model.L/2 - 1 )/model.Λ_U[d] - model.β - 1/2*sum( model.U[d,:].^2 )
        dlogΛ_V[d] = ( model.α + model.M/2 - 1 )/model.Λ_U[d] - model.β - 1/2*sum( model.V[d,:].^2 )
    end
    return ( dlogΛ_U, dlogΛ_V )
end
