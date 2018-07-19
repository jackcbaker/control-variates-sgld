using Distributions
include("MatFac.jl")


"""
Container for sgld parameters
"""
type saga
    subsize::Int64                  # Minibatch size
    iter::Int64
    subsample::Array{ Int64, 1 }
    ϵ_U::Float64             # Stepsize tuning constants
    G_U::Float64
    g_alphaU_i::Array{Float64,2}
    g_alphaU::Array{Float64,2}
    ϵ_V::Float64             # Stepsize tuning constants
    G_V::Float64
    g_alphaV_i::Array{Float64,2}
    g_alphaV::Array{Float64,2}
    ϵ_a::Float64             # Stepsize tuning constants
    G_a::Float64
    g_alphaa_i::Array{Float64,1}
    g_alphaa::Array{Float64,1}
    ϵ_b::Float64             # Stepsize tuning constants
    G_b::Float64
    g_alphab_i::Array{Float64,1}
    g_alphab::Array{Float64,1}
end

"""
Set standard tuning values
"""
function saga( model::MatFac, subsize::Int64, opt_stepsize::Float64 )
    subsample = sample( 1:model.N, subsize )
    ϵ_U = opt_stepsize
    ϵ_V = opt_stepsize
    ϵ_a = opt_stepsize
    ϵ_b = opt_stepsize
    G_U = 0
    G_V = 0
    G_a = 0
    G_b = 0
    # returns tuple of g_alpha for each parameter to keep things tidy
    g_alpha_i = dlogdens( model, 1:model.N )
    ( g_alphaU, g_alphaV, g_alphaa, g_alphab ) = loglikests( model, g_alpha_i, subsample )
    # Expand g_alpha tuple
    ( g_alphaU_i, g_alphaV_i, g_alphaa_i, g_alphab_i ) = g_alpha_i
    saga( subsize, 1, subsample, ϵ_U, G_U, g_alphaU_i, g_alphaU, ϵ_V, G_V, g_alphaV_i, g_alphaV, 
            ϵ_a, G_a, g_alphaa_i, g_alphaa, ϵ_b, G_b, g_alphab_i, g_alphab )
end

"""
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function saga_update( model::MatFac, tuning::saga )
    # Subsample entries
    tuning.subsample = sample( 1:model.N, tuning.subsize )
    # Calculate log density estimates (returned as tuple for brevity)
    g_alpha_new = dlogdens( model, tuning.subsample )
    # Get current alpha gradient estimates for subsample
    g_alpha_curr = get_alpha_curr( tuning )
    # Get new and current log likelihood estimates at alpha
    dloglik_alpha = loglikests( model, g_alpha_curr, tuning.subsample )
    dloglik_new = loglikests( model, g_alpha_new, tuning.subsample )
    # Calculate SAGA estimate of log posterior gradient
    ( dlogU, dlogV, dloga, dlogb ) = dlogpostsaga( model, tuning, dloglik_alpha, dloglik_new )
    # Update storage of gradients at alpha
    update_g_alpha( tuning, g_alpha_new, dloglik_alpha, dloglik_new )

    # Update using langevin dynamics using SAGA log posterior estimates
    ζ = 1 / tuning.iter
    ξ = tuning.iter^(-0.33)
    ϵ_U = tuning.ϵ_U * ξ
    tuning.G_U += ζ*( mean( dlogU .^ 2 ) - tuning.G_U )
    η = sqrt( ϵ_U / sqrt( tuning.G_U ) ) * rand( Normal( 0, 1 ), ( model.D, model.L ) )
    model.U += ϵ_U / ( 2 * sqrt( tuning.G_U ) ) * dlogU + η
    if tuning.iter % 10 == 0
        println(sum(dlogU))
    end

    ϵ_V = tuning.ϵ_V * ξ
    tuning.G_V += ζ*( mean( dlogV .^ 2 ) - tuning.G_V )
    η = sqrt( ϵ_V / sqrt( tuning.G_V ) ) * rand( Normal( 0, 1 ), ( model.D, model.M ) )
    model.V += ϵ_V / ( 2 * sqrt( tuning.G_V ) ) * dlogV + η

    ϵ_a = tuning.ϵ_a * ξ
    tuning.G_a += ζ*( mean( dloga .^ 2 ) - tuning.G_a )
    η = sqrt( ϵ_a / sqrt( tuning.G_a ) ) * rand( Normal( 0, 1 ), model.L )
    model.a += ϵ_a / ( 2 * sqrt( tuning.G_a ) ) * dloga + η

    ϵ_b = tuning.ϵ_b * ξ
    tuning.G_b += ζ*( mean( dlogb .^ 2 ) - tuning.G_b )
    η = sqrt( ϵ_b / sqrt( tuning.G_b ) ) * rand( Normal( 0, 1 ), model.M )
    model.b += ϵ_b / ( 2 * sqrt( tuning.G_b ) ) * dlogb + η
    
    # Update hyperparameters using Gibbs step every 100 iterations
    if tuning.iter % 10 == 0
        update_Λ( model )
    end
end


"""
Calculate log likelihood estimates from vector of gradients
"""
function loglikests( model::MatFac, g_alpha_i::Tuple, subsample )
    # Expand g_alpha_i tuple
    ( g_alphaU_i, g_alphaV_i, g_alphaa_i, g_alphab_i ) = g_alpha_i
    # Initialise containers for gradients
    dloglikU = zeros( model.D, model.L )
    dloglikV = zeros( model.D, model.M )
    dloglika = zeros( model.L )
    dloglikb = zeros( model.M )
    for (i,index) in enumerate(subsample)
        ( user, item, rating ) = model.train[index,:]
        dloglikU[:,user] += g_alphaU_i[i,:]
        dloglikV[:,item] += g_alphaV_i[i,:]
        dloglika[user] += g_alphaa_i[i]
        dloglikb[item] += g_alphab_i[i]
    end
    return ( dloglikU, dloglikV, dloglika, dloglikb )
end


"""
Get current gradient estimates at alpha for minibatch
"""
function get_alpha_curr( tuning::saga )
    g_alphaU_curr = tuning.g_alphaU_i[tuning.subsample,:]
    g_alphaV_curr = tuning.g_alphaV_i[tuning.subsample,:]
    g_alphaa_curr = tuning.g_alphaa_i[tuning.subsample]
    g_alphab_curr = tuning.g_alphab_i[tuning.subsample]
    return ( g_alphaU_curr, g_alphaV_curr, g_alphaa_curr, g_alphab_curr )
end


"""
Update g_alpha with current minibatch
"""
function update_g_alpha( tuning::saga, g_alpha_new::Tuple, dloglik_alpha::Tuple, dloglik_new::Tuple )
    # Expand log likelihood tuples
    ( dloglikU_new, dloglikV_new, dloglika_new, dloglikb_new ) = dloglik_new
    ( dloglikU_alpha, dloglikV_alpha, dloglika_alpha, dloglikb_alpha ) = dloglik_alpha
    # Update full log likelihood gradient estimates
    tuning.g_alphaU += dloglikU_new - dloglikU_alpha
    tuning.g_alphaV += dloglikV_new - dloglikV_alpha
    tuning.g_alphaa += dloglika_new - dloglika_alpha
    tuning.g_alphab += dloglikb_new - dloglikb_alpha
    # Expand g_alpha tuple
    ( g_alphaU_new, g_alphaV_new, g_alphaa_new, g_alphab_new ) = g_alpha_new
    # Update gradient storage
    tuning.g_alphaU_i[tuning.subsample,:] = g_alphaU_new
    tuning.g_alphaV_i[tuning.subsample,:] = g_alphaV_new
    tuning.g_alphaa_i[tuning.subsample] = g_alphaa_new
    tuning.g_alphab_i[tuning.subsample] = g_alphab_new
end


"""
Calculate SAGA log posterior estimate
"""
function dlogpostsaga( model::MatFac, tuning::saga, dloglik_alpha::Tuple, dloglik_new::Tuple )
    # Calculate correction for subsampling
    correction = model.N / tuning.subsize
    # Expand log likelihood tuples
    ( dloglikU_new, dloglikV_new, dloglika_new, dloglikb_new ) = dloglik_new
    ( dloglikU_alpha, dloglikV_alpha, dloglika_alpha, dloglikb_alpha ) = dloglik_alpha
    # Calculate log posterior estimates
    minibatch = model.train[tuning.subsample,:]
    dlogpostU = tuning.g_alphaU + correction * ( dloglikU_new - dloglikU_alpha )
    dlogpriorU( model, minibatch, dlogpostU )
    dlogpostV = tuning.g_alphaV + correction * ( dloglikV_new - dloglikV_alpha )
    dlogpriorV( model, minibatch, dlogpostV )
    dlogposta = tuning.g_alphaa + correction * ( dloglika_new - dloglika_alpha )
    dlogpriora( model, minibatch, dlogposta )
    dlogpostb = tuning.g_alphab + correction * ( dloglikb_new - dloglikb_alpha )
    dlogpriorb( model, minibatch, dlogpostb )
    return ( dlogpostU, dlogpostV, dlogposta, dlogpostb )
end
