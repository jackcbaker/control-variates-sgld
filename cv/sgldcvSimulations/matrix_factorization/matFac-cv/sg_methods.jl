using Distributions
include("./models/control_variates_bpmf.jl")

"""
Update one step of Stochastic Gradient Langevin Dynamics for a Latent Dirichlet Allocation model
"""
function sgld_update( model::matrix_factorisation, tuning::sgld, cv::control_variate )
    
    # Subsample documents
    tuning.subsample = sample( 1:model.N, tuning.subsize )

    # Simulate Langevin dynamics of new submodel
    cv_update( model, cv, tuning )
    
    # Update hyperparameters using Gibbs step every 100 iterations
    if tuning.iter % 10 == 0
        update_Λ( model )
    end

    return( model )
end

"""
Update one step of stochastic gradient descent
"""
function sgd_update( model::matrix_factorisation, tuning::sgd )
    tuning.subsample = sample( 1:model.N, tuning.subsize )
    γ = 1 / ( 1 + tuning.iter )
    ζ = 1 / ( 1 + tuning.iter )    
    ( dlogU, dlogV, dloga, dlogb ) = dlogpost( model, tuning.subsample, model.U, model.V, 
                                               model.a, model.b )
    tuning.G_U += ζ*( mean(dlogU.^2) - tuning.G_U )
    model.U += tuning.ϵ_U / 2 * dlogU / sqrt( tuning.G_U )
    tuning.G_V += ζ*( mean(dlogV.^2) - tuning.G_V )
    model.V += tuning.ϵ_V / 2 * dlogV / sqrt( tuning.G_V )
    tuning.G_a += ζ*( mean(dloga.^2) - tuning.G_a )
    model.a += tuning.ϵ_a / 2 * dloga / sqrt( tuning.G_a )
    tuning.G_b += ζ*( mean(dlogb.^2) - tuning.G_b )
    model.b += tuning.ϵ_b / 2 * dlogb / sqrt( tuning.G_b )
    tuning.U = model.U
    tuning.V = model.V
    tuning.a = model.a
    tuning.b = model.b
    if tuning.iter % 10 == 0
        update_Λ( model )
    end
end


"""
Calculate full grad log post at mode
"""
function grad_log_post( model::matrix_factorisation, tuning::sgd )
    println( "Calculating full gradlogpost..." )
    ( dlogU, dlogV, dloga, dlogb ) = dlogpostfull( model, model.U, model.V, 
                                               model.a, model.b )
    tuning.ll_U = dlogU
    tuning.ll_V = dlogV
    tuning.ll_a = dloga
    tuning.ll_b = dlogb
end
