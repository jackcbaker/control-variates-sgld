include( "sg_methods.jl" )
using JLD

"""
Simulate from an LDA model using SGLD
"""
function tune_sgld( model::lda, step_const::Float64, subsize::Int64, n_iter::Int64 )

    # Generate objects and storage
    tuning = sgld( model, step_const, subsize )
    println( tuning.step_const )
    mkpath("perplex_out/std/$(model.M)")

    # Simulate from model using SGLD
    println("Sampling from posterior...")
    tic()
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            perplex = perplexity( model, model.test, model.theta )
            iter_time = toq()
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/std/$(model.M)/out-$(step_const).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            tic()
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
        end
        sgld_update( model, tuning )
    end
end

"""
Simulate from an LDA model using SGLD
"""
function sim_sgld( model::lda, step_const::Float64, subsize::Int64, n_iter::Int64, seed::Int64 )

    # Generate objects and storage
    tuning = sgld( model, step_const, subsize )
    println( tuning.step_const )
    mkpath("perplex_out/std/$(model.M)")

    # Simulate from model using SGLD
    println("Sampling from posterior...")
    tic()
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            iter_time = toq()
            perplex = perplexity( model, model.test, model.theta )
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/std/$(model.M)/out-$(seed).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            tic()
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
        end
        sgld_update( model, tuning )
    end
end
