include( "saga.jl" )
using JLD

"""
Simulate from an LDA model using SGLD
"""
function sim_saga( model::lda, epsilon::Float64, subsize::Int64, n_iter::Int64 )

    # Generate objects and storage
    tuning = saga( model, epsilon, subsize )
    println( tuning.epsilon )
    mkpath("perplex_out/$(model.M)")

    # Simulate from model using SGLD
    println("Sampling from posterior...")
    tic()
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            perplex = perplexity( model, model.test, model.theta )
            iter_time = toq()
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/$(model.M)/out-$(epsilon).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
            tic()
        end
        saga_update( model, tuning )
    end
end


"""
Simulate from an LDA model using SGLD
"""
function sim_saga( model::lda, epsilon::Float64, subsize::Int64, n_iter::Int64, seed::Int64 )

    # Generate objects and storage
    tuning = saga( model, epsilon, subsize )
    println( tuning.epsilon )
    mkpath("perplex_out/$(model.M)")

    # Simulate from model using SGLD
    println("Sampling from posterior...")
    tic()
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            perplex = perplexity( model, model.test, model.theta )
            iter_time = toq()
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/$(model.M)/out-$(seed).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
            tic()
        end
        saga_update( model, tuning )
    end
end
