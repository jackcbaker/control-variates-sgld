using StatsBase
using Distributions
include("sgd.jl")

"""
Container for sgld parameters
"""
type sgld
    subsize::Int64                  # Minibatch size
    iter::Int64
    subsample::Array{ Int64, 1 }
    permuted_data::Array{ Int64, 1 }
    array_index::Int64
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
    permuted_data = randperm(model.M)
    array_index = 1
    ϵ_U = opt_stepsize
    ϵ_V = opt_stepsize
    ϵ_a = opt_stepsize
    ϵ_b = opt_stepsize
    G_U = 0
    G_V = 0
    G_a = 0
    G_b = 0
    sgld( subsize, 1, subsample, permuted_data, array_index, ϵ_U, G_U, ϵ_V, G_V, ϵ_a, G_a, ϵ_b, G_b )
end

"""
Control variate object for BPMF example
"""
type control_variate
    η_U::Float64
    η_V::Float64
    η_a::Float64
    η_b::Float64
    U::Array{ Float64, 2 }
    ll_U::Array{ Float64, 2 }
    G_U::Array{ Float64, 2 }
    V::Array{ Float64, 2 }
    ll_V::Array{ Float64, 2 }
    G_V::Array{ Float64, 2 }
    a::Array{ Float64, 1 }
    ll_a::Array{ Float64, 1 }
    G_a::Array{ Float64, 1 }
    b::Array{ Float64, 1 }
    ll_b::Array{ Float64, 1 }
    G_b::Array{ Float64, 1 }
end

function control_variate( model::matrix_factorisation, sgd_init::sgd )
    println("Calculating full posterior gradient...")
    println("Done")
    η_U = sgd_init.ϵ_U
    η_V = sgd_init.ϵ_V    
    η_a = sgd_init.ϵ_a
    η_b = sgd_init.ϵ_b
    U = sgd_init.U
    ll_U = sgd_init.ll_U
    G_U = sgd_init.G_U
    V = sgd_init.V
    ll_V = sgd_init.ll_V
    G_V = sgd_init.G_V
    a = sgd_init.a
    ll_a = sgd_init.ll_a
    G_a = sgd_init.G_a
    b = sgd_init.b
    ll_b = sgd_init.ll_b
    G_b = sgd_init.G_b
    control_variate( η_U, η_V, η_a, η_b, U, ll_U, G_U, V, ll_V, G_V, a, ll_a, G_a, b, ll_b, G_b )
end

function rmse( model::matrix_factorisation, cv::control_variate )
    rmse = 0
    test_size = size( model.test, 1 )
    for i in 1:test_size
        ( user, item, rating ) = model.test[i,:]
        try
            rmse += ( rating - dot( cv.U[:,user], cv.V[:,item] - 
                      cv.a[user] - cv.b[item] ) )^2
        catch
            continue
        end
    end
    rmse = sqrt( rmse / test_size )
    return rmse
end

function cv_update( model::matrix_factorisation, cv::control_variate, tuning::sgld )
    ζ = 1 / tuning.iter
    # Calculate required gradient estimates
    ( dlogU, dlogV, dloga, dlogb ) = dlogpost( model, tuning.subsample, model.U, model.V, 
                                               model.a, model.b )
    ( dlogUopt, dlogVopt, dlogaopt, dlogbopt ) = dlogpost( model, tuning.subsample, cv.U, cv.V, 
                                                           cv.a, cv.b )
    # Update U
    tuning.G_U += ζ*( mean( ( cv.ll_U - dlogUopt + dlogU ) .^ 2 ) - tuning.G_U )
    η = sqrt( tuning.ϵ_U / sqrt( tuning.G_U ) ) * rand( Normal( 0, 1 ), ( model.D, model.L ) )
    model.U += tuning.ϵ_U / ( sqrt( tuning.G_U ) * 2 ) * ( cv.ll_U - dlogUopt + dlogU ) + η
    # Update V
    tuning.G_V += ζ*( mean( ( cv.ll_V - dlogVopt + dlogV ) .^ 2 ) - tuning.G_V )
    η = sqrt( tuning.ϵ_V / sqrt( tuning.G_V ) ) * rand( Normal( 0, 1 ), ( model.D, model.M ) )
    model.V += tuning.ϵ_V / ( sqrt( tuning.G_V ) * 2 ) * ( cv.ll_V - dlogVopt + dlogV ) + η
    # Update a
    tuning.G_a += ζ*( mean( ( cv.ll_a - dlogaopt + dloga ) .^ 2 ) - tuning.G_a )
    η = sqrt( tuning.ϵ_a / sqrt( tuning.G_a ) ) * rand( Normal( 0, 1 ), model.L )
    model.a += tuning.ϵ_a / ( sqrt( tuning.G_a ) * 2 ) * ( cv.ll_a - dlogaopt + dloga ) + η
    # Update b
    tuning.G_b += ζ*( mean( ( cv.ll_b - dlogbopt + dlogb ) .^ 2 ) - tuning.G_b )
    η = sqrt( tuning.ϵ_b / sqrt( tuning.G_b ) ) * rand( Normal( 0, 1 ), model.M )
    model.b += tuning.ϵ_b / ( sqrt( tuning.G_b ) * 2 ) * ( cv.ll_b - dlogbopt + dlogb ) + η

    model.dlogU = ( cv.ll_U - dlogUopt + dlogU )
    model.dlogV = ( cv.ll_V - dlogVopt + dlogV )
    model.dloga = ( cv.ll_a - dlogaopt + dloga )
    model.dlogb = ( cv.ll_b - dlogbopt + dlogb )
end
