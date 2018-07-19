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
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function sgld_update( model::lda, tuning::sgld )
    
    # Subsample documents
    tuning.subsample = sample( 1:model.M, tuning.subsize )

    # Simulate Langevin dynamics of new submodel
    epsilon = tuning.step_const
    η = sqrt(epsilon) * rand( Normal(), ( model.K, model.V ) )
    model.zcounts[tuning.subsample,:,:] = update_topics( model, tuning.subsample )
    # Topic word parameter
    model.theta += epsilon/2 * dlogpost( model, tuning.subsample ) + η

    return( model )
end
