using JLD
include( "saga.jl" )

"""
Simulate from an LDA model using SGLD
"""
function tune_saga( model::MatFac, stepsize::Float64, subsize::Int64, n_iter::Int64 )

    # Generate objects and storage
    tuning = saga( model, subsize, stepsize )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$(tuning.ϵ_U).log", force=true )
    
    tic()
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            iter_time = toq()
            println("$(tuning.iter)\t$rmse_current\t$iter_time")
            open("rmse_out/$(model.L)/out-$(tuning.ϵ_U).log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        saga_update( model, tuning )
    end
end

"""
Simulate from an LDA model using SAGA
"""
function sim_saga(model::MatFac, stepsize::Float64, seed_curr::Int64, subsize::Int64, n_iter::Int64)

    # Generate objects and storage
    tuning = saga( model, subsize, stepsize )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$(seed_curr).log", force=true )
    
    tic()
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            iter_time = toq()
            println("$(tuning.iter)\t$rmse_current\t$iter_time")
            open("rmse_out/$(model.L)/out-$(seed_curr).log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        saga_update( model, tuning )
    end
end
