using Distributions
include("lda.jl")

"""
Container for sgld parameters
"""
type saga
    subsize::Int64                  # Minibatch size
    epsilon::Float64             # Stepsize tuning constant
    iter::Int64
    subsample::Array{Int64,1}
    g_alpha_i::Array{Float64,3}
    g_alpha::Array{Float64,2}
end

"""
Set standard tuning values
"""
function saga( model::lda, epsilon, subsize = 200::Int64 )
    iter = 1
    subsample = [-1]
    g_alpha_i = dlogdens( model, 1:model.M )
    g_alpha = squeeze( sum( g_alpha_i, 1 ), 1 )
    saga( subsize, epsilon, iter, subsample, g_alpha_i, g_alpha )
end

"""
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function saga_update( model::lda, tuning::saga )
    # Subsample documents
    tuning.subsample = sample( 1:model.M, tuning.subsize )
    # Calculate log density gradients for minibatch wrt theta
    dlogdens_theta = dlogdens( model, tuning.subsample )
    # Calculate minibatch log likelihood gradient estimates at theta and at alpha
    dloglik_theta = squeeze( sum( dlogdens_theta, 1 ), 1 )
    dloglik_alpha = squeeze( sum( tuning.g_alpha_i[tuning.subsample,:,:], 1 ), 1 )
    # Calculate SAGA estimate of log posterior gradient
    dlogpostest = dlogpostsaga( model, tuning, dloglik_theta, dloglik_alpha )
    if tuning.iter % 10 == 0
        println(sum(dlogpostest))
    end

    # Update g_alpha
    tuning.g_alpha += dloglik_theta - dloglik_alpha
    tuning.g_alpha_i[tuning.subsample,:,:] = dlogdens_theta

    # Gibbs update of z topic allocations
    model.zcounts[tuning.subsample,:,:] = update_topics( model, tuning.subsample )
    # Simulate Langevin dynamics using SAGA log posterior estimate
    injected_noise = Normal( 0, 1 )
    model.theta += tuning.epsilon/2 * dlogpostest
    model.theta += sqrt(tuning.epsilon) * rand( injected_noise, ( model.K, model.V )  )
end

"""
Calculate SAGA log posterior estimate at current state model.theta
"""
function dlogpostsaga( model::lda, tuning::saga, dloglik_theta::Array{Float64,2}, 
        dloglik_alpha::Array{Float64,2} )
    correction = model.M / tuning.subsize
    dlogpostest = tuning.g_alpha + correction * ( dloglik_theta - dloglik_alpha )
    dlogpostest += dlogprior( model )
    return dlogpostest
end
