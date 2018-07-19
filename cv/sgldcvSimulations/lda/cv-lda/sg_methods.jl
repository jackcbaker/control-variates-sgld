using Distributions
include("./models/sgd.jl")

"""
Container for sgld parameters
"""
type sgld
    subsize::Int64                  # Minibatch size
    step_const::Float64             # Stepsize tuning constant
    iter::Int64
    subsample::Array{ Int64, 1 }
    permuted_data::Array{ Int64, 1 }
    array_index::Int64
end

"""
Set standard tuning values
"""
function sgld( model::lda, step_const, subsize = 200::Int64 )
    iter = 1
    subsample = [-1]
    permuted_data = randperm(model.M)
    array_index = 1
    sgld( subsize, step_const, iter, subsample, permuted_data, array_index )
end

"""
Container for control variate states
"""
type control_variate
    theta::Array{ Float64, 2 }
    pe_opt::Array{ Float64, 2 }
    update::Array{ Float64, 2 }
end

"""
Standard initialisation for control variate container
"""
function control_variate( model::lda, graddesc::sgd )
    theta = model.theta
    pe_opt = graddesc.ll
    update = zeros( size( pe_opt ) )
    control_variate( theta, pe_opt, update )
end

"""
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function sgld_update( model::lda, tuning::sgld, cv::control_variate )
    
    # Subsample documents
    tuning.subsample = sample( 1:model.M, tuning.subsize )

    # Simulate Langevin dynamics of new submodel
    epsilon = tuning.step_const
    model.zcounts[tuning.subsample,:,:] = update_topics( model, tuning.subsample )
    cv_update( model, tuning, cv )
    # Topic word parameter
    η = sqrt(epsilon) * rand( Normal(), ( model.K, model.V ) )
    model.theta += epsilon/2 * cv.update + η

    return( model )
end

"""
Adaptively update control variate parameters
"""
function cv_update( model::lda, tuning::sgld, cv::control_variate )
    pe_est = dlogpost( model, tuning.subsample )
    pe_opt_est = pe(  model, tuning.subsample, cv.theta )
    cv.update = cv.pe_opt - ( pe_opt_est - pe_est )
end
