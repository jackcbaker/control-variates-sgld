using JLD
include( "sg_methods.jl" )

function sgld_sgd_init( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, subsize::Int64, n_iter::Int64 )
    # Find good initial values for SGLD using SGD
    sgd_init = load( "./sgd_out/$(model.L)/sgd_init-$sgd_step.jld" )["sgd_init"]
    model.U = sgd_init.U
    model.V = sgd_init.V
    model.a = sgd_init.a
    model.b = sgd_init.b
    update_Λ( model )
#    run_sgd( model, sgd_step, subsize, 1000 )
    # Run SGLD after SGD run
    sim_sgld( model, stepsize, sgd_init, subsize, n_iter, sgd_step )
end

"""
Simulate from an LDA model using SGLD
"""
function sim_sgld( model::matrix_factorisation, stepsize::Float64, sgd_init::sgd, subsize::Int64, n_iter::Int64, label::Float64 )

    # Generate objects and storage
    println("Number of users: $(model.L)\tNumber of items: $(model.M)")
    tuning = sgld( model, subsize, stepsize )
    cv = control_variate( model, sgd_init )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$(tuning.ϵ_U)-$label.log", force=true )
    tic()
    
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            iter_time = toq()
            rmse_opt = rmse( model, cv )
            println("$(tuning.iter)\t$rmse_current\t$rmse_opt\t$iter_time")
            open("rmse_out/$(model.L)/out-$(tuning.ϵ_U)-$label.log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$rmse_opt\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        # Update 1 iteration
        sgld_update( model, tuning, cv )
    end
end

function run_sgd( model::matrix_factorisation, stepsize::Float64, subsize::Int64, n_iter::Int64 )
    tuning = sgd( model, subsize, stepsize )
    mkpath("sgd_log/$(model.L)/")
    rm("sgd_log/model.L/out-$stepsize.log", force=true )
    tic()

    for tuning.iter in 1:n_iter
        if tuning.iter % 10 == 0
            rmse_current = rmse( model )
            elapsed = toq()
            println("$(tuning.iter)\t$rmse_current\t$elapsed")
            open("sgd_log/$(model.L)/out-$stepsize.log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$elapsed\n")
            end
            tic()
        end
        sgd_update( model, tuning )
    end
    grad_log_post( model, tuning )
    return( tuning )
end


function sgld_final( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, subsize::Int64, n_iter::Int64, seed::Int64 )
    # Find good initial values for SGLD using SGD
    sgd_init = load( "./sgd_out/$(model.L)/sgd_init-$sgd_step.jld" )["sgd_init"]
    model.U = sgd_init.U
    model.V = sgd_init.V
    model.a = sgd_init.a
    model.b = sgd_init.b
    update_Λ( model )
#    run_sgd( model, sgd_step, subsize, 1000 )
    # Run SGLD after SGD run
    sim_final( model, stepsize, sgd_init, subsize, n_iter, seed )
end

"""
Simulate from an LDA model using SGLD
"""
function sim_final( model::matrix_factorisation, stepsize::Float64, sgd_init::sgd, subsize::Int64, n_iter::Int64, seed::Int64 )

    # Generate objects and storage
    println("Number of users: $(model.L)\tNumber of items: $(model.M)")
    tuning = sgld( model, subsize, stepsize )
    cv = control_variate( model, sgd_init )
    mkpath("rmse_out/$(model.L)/")
    rm("rmse_out/model.L/out-$(seed).log", force=true )
    tic()
    
    # Simulate using SGLD
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse( model )
            iter_time = toq()
            rmse_opt = rmse( model, cv )
            println("$(tuning.iter)\t$rmse_current\t$rmse_opt\t$iter_time")
            open("rmse_out/$(model.L)/out-$(seed).log", "a") do f
                write(f, "$(tuning.iter)\t$rmse_current\t$rmse_opt\t$iter_time\n")
            end
            tic()
            if ( ( sum( isnan( model.U ) ) > 0 ) | ( sum( isnan( model.U ) ) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        # Update 1 iteration
        sgld_update( model, tuning, cv )
    end

    return( rmse_storage )
end
