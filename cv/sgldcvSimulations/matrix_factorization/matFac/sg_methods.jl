using Distributions
include("./models/sgd.jl")

"""
Container for sgld parameters
"""
type sgld
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
end

"""
Set standard tuning values
"""
function sgld( model::matrix_factorisation, subsize::Int64, opt_stepsize::Float64 )
    subsample = sample( 1:model.N, subsize )
    ϵ_U = opt_stepsize
    ϵ_V = opt_stepsize
    ϵ_a = opt_stepsize
    ϵ_b = opt_stepsize
    G_U = 0
    G_V = 0
    G_a = 0
    G_b = 0
    sgld( subsize, 1, subsample, ϵ_U, G_U, ϵ_V, G_V, ϵ_a, G_a, ϵ_b, G_b )
end

"""
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function sgld_update( model::matrix_factorisation, tuning::sgld )
    
    # Subsample documents
    tuning.subsample = sample( 1:model.N, tuning.subsize )
    ζ = 1 / tuning.iter 

    # Simulate Langevin dynamics of new submodel
    ( dlogU, dlogV, dloga, dlogb ) = dlogpost( model, tuning.subsample, model.U, model.V, 
                                               model.a, model.b )
    tuning.G_U += ζ*( mean( dlogU .^ 2 ) - tuning.G_U )
    η = sqrt( tuning.ϵ_U / sqrt( tuning.G_U ) ) * rand( Normal( 0, 1 ), ( model.D, model.L ) )
    model.U += tuning.ϵ_U / ( 2 * sqrt( tuning.G_U ) ) * dlogU + η
    if tuning.iter % 10 == 0
        println(sum(dlogU))
    end

    tuning.G_V += ζ*( mean( dlogV .^ 2 ) - tuning.G_V )
    η = sqrt( tuning.ϵ_V / sqrt( tuning.G_V ) ) * rand( Normal( 0, 1 ), ( model.D, model.M ) )
    model.V += tuning.ϵ_V / ( 2 * sqrt( tuning.G_V ) ) * dlogV + η

    tuning.G_a += ζ*( mean( dloga .^ 2 ) - tuning.G_a )
    η = sqrt( tuning.ϵ_a / sqrt( tuning.G_a ) ) * rand( Normal( 0, 1 ), model.L )
    model.a += tuning.ϵ_a / ( 2 * sqrt( tuning.G_a ) ) * dloga + η

    tuning.G_b += ζ*( mean( dlogb .^ 2 ) - tuning.G_b )
    η = sqrt( tuning.ϵ_b / sqrt( tuning.G_b ) ) * rand( Normal( 0, 1 ), model.M )
    model.b += tuning.ϵ_b / ( 2 * sqrt( tuning.G_b ) ) * dlogb + η
    
    # Update hyperparameters using Gibbs step every 100 iterations
    if tuning.iter % 10 == 0
        update_Λ( model )
    end
end

"""
Update one step of stochastic gradient descent
"""
function sgd_update( model::matrix_factorisation, tuning::sgd )
    tuning.subsample = sample( 1:model.N, tuning.subsize )
    ζ = 0.1
    ( dlogU, dlogV, dloga, dlogb ) = dlogpost( model, tuning.subsample, model.U, model.V, 
                                               model.a, model.b )
    tuning.G_U += ζ*( dlogU.^2 - tuning.G_U )
    model.U += tuning.ϵ_U / 2 * dlogU ./ sqrt( tuning.G_U )
    tuning.G_V += ζ*( dlogV.^2 - tuning.G_V )
    model.V += tuning.ϵ_V / 2 * dlogV ./ sqrt( tuning.G_V )
    tuning.G_a += ζ*( dloga.^2 - tuning.G_a )
    model.a += tuning.ϵ_a / 2 * dloga ./ sqrt( tuning.G_a )
    tuning.G_b += ζ*( dlogb.^2 - tuning.G_b )
    model.b += tuning.ϵ_b / 2 * dlogb ./ sqrt( tuning.G_b )
    if tuning.iter % 10 == 0
        update_Λ( model )
    end
end
