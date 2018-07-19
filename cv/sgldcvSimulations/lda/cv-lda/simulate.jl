include( "sg_methods.jl" )
using JLD


function sgld_sgd_init( model::lda, stepsize::Float64, sgd_step::Float64, subsize::Int64, n_iter::Int64, seed::Int64 )
    # Load estimates for log likelihood and theta hat from SGD run
    println( "SGD step: $sgd_step" )
    println( model.M )
    println( "Running initial stochastic gradient descent" )
    sgd_init = run_sgd( model, sgd_step, subsize, 20000, seed )
    println( "Sampling using SGLDCV" )
    sgld_final( model, stepsize, sgd_init, subsize, 20000, seed )
end

"""
Simulate from an LDA model using SGLD
"""
function sim_sgld( model::lda, step_const::Float64, sgd_init::sgd, subsize::Int64, n_iter::Int64 )

    # Generate objects
    tuning = sgld( model, step_const, subsize )
    println( tuning.step_const )
    cv = control_variate( model, sgd_init )
    mkpath("perplex_out/cv/$(model.M)/")
    rm("perplex_out/cv/$(model.M)/out-$(step_const).log", force=true )
    # Storage
    avg_theta = zeros( size( model.theta ) )

    tic()
    # Simulate from model using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            perplex = perplexity( model, model.test, avg_theta )
            println( perplexity( model, model.test ) )
            iter_time = toq()
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/cv/$(model.M)/out-$(step_const).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            tic()
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
        end
        sgld_update( model, tuning, cv )
    end
end

function run_sgd( model::lda, stepsize::Float64, subsize::Int64, n_iter::Int64, seed::Int64 )
    tuning = sgd( model, subsize, stepsize )
    mkpath("sgd_out/$(model.M)")
    rm("sgd_out/$(model.M)/out-$seed.log", force=true )
    tic()
    for tuning.iter in 1:n_iter
        if tuning.iter % 10 == 0
            iter_time = toq()
            perplex_current = perplexity( model, model.test )
            println("$(tuning.iter)\t$perplex_current\t$iter_time")
            open("sgd_out/$(model.M)/out-$seed.log", "a") do f
                write(f, "$(tuning.iter)\t$perplex_current\t$iter_time\n")
            end
            tic()
        end
        sgd_update( model, tuning )
    end
    tuning.ll = pe( model, collect(1:model.M), tuning.theta )
    return( tuning )
end

"""
Simulate from an LDA model using SGLD
"""
function sgld_final( model::lda, step_const::Float64, sgd_init::sgd, subsize::Int64, n_iter::Int64, seed::Int64 )

    # Generate objects
    tuning = sgld( model, step_const, subsize )
    println( tuning.step_const )
    cv = control_variate( model, sgd_init )
    mkpath("perplex_out/cv/$(model.M)/")
    rm("perplex_out/cv/$(model.M)/out-$(seed).log", force=true )

    tic()
    # Simulate from model using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            iter_time = toq()
            perplex = perplexity( model, model.test )
            println("$(tuning.iter)\t$perplex\t$iter_time")
            open("perplex_out/cv/$(model.M)/out-$(seed).log", "a") do f
                write(f, "$(tuning.iter)\t$perplex\t$iter_time\n")
            end
            tic()
            if ( sum( isnan( model.theta ) ) > 0 )
                print("\n")
                error("The chain has diverged")
            end
        end
        sgld_update( model, tuning, cv )
    end
end
