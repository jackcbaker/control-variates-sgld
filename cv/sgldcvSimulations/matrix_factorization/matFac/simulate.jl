using JLD
include( "sg_methods.jl" )

"""
Simulate from an LDA model using SGLD
"""
function sim_sgld( model::matrix_factorisation, stepsize::Float64, subsize::Int64, n_iter::Int64 )

    # Generate objects and storage
    tuning = sgld( model, subsize, stepsize )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$(tuning.ϵ_U).log", force=true )
    tic()
    
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            prop_processed = ( tuning.iter * tuning.subsize ) / model.N
            iter_time = toq()
            println("iter: $(tuning.iter) rmse: $rmse_current time: $iter_time")
            open("rmse_out/$(model.L)/out-$(tuning.ϵ_U).log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        # Update 1 iteration
        sgld_update( model, tuning )
    end

    return( rmse_storage )
end


"""
Simulate from an LDA model using SGLD
"""
function sample_sgld( model::matrix_factorisation, stepsize::Float64, subsize::Int64, n_iter::Int64, seed_current::Int64 )

    # Generate objects and storage
    println( "Running for $n_iter iterations..." )
    tuning = sgld( model, subsize, stepsize )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$seed_current.log", force=true )
    tic()
    
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            iter_time = toq()
            println("iter: $(tuning.iter) rmse: $rmse_current time: $iter_time")
            open("rmse_out/$(model.L)/out-$seed_current.log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        # Update 1 iteration
        sgld_update( model, tuning )
    end
end
