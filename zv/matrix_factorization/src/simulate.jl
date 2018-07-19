using JLD
using PyCall
@pyimport sklearn.covariance as skov
include( "sg_methods.jl" )

"""
Storage object
"""
type Storage
    U::Array{ Float64, 3 }              # Latent feature vector
    V::Array{ Float64, 3 }              # Latent feature vector
    a::Array{ Float64, 2 }              # User bias terms
    b::Array{ Float64, 2 }              # Item bias terms
    dlogU::Array{ Float64, 3 }
    dlogV::Array{ Float64, 3 }
    dloga::Array{ Float64, 2 }
    dlogb::Array{ Float64, 2 }
end

function Storage( model, n_iter )
    U = zeros( ( n_iter, model.D, model.L ) )
    dlogU = zeros( ( n_iter, model.D, model.L ) )
    V = zeros( ( n_iter, model.D, model.M ) )
    dlogV = zeros( ( n_iter, model.D, model.M ) )
    a = zeros( n_iter, model.L )
    dloga = zeros( n_iter, model.L )
    b = zeros( n_iter, model.M )
    dlogb = zeros( n_iter, model.M )
    Storage( U, V, a, b, dlogU, dlogV, dloga, dlogb )
end

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
    stored = Storage( model, n_iter )
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
        # Update storage
        stored.U[tuning.iter,:,:] = model.U
        stored.dlogU[tuning.iter,:,:] = model.dlogU
        stored.V[tuning.iter,:,:] = model.V
        stored.dlogV[tuning.iter,:,:] = model.dlogV
        stored.a[tuning.iter,:] = model.a
        stored.dloga[tuning.iter,:] = model.dloga
        stored.b[tuning.iter,:] = model.b
        stored.dlogb[tuning.iter,:] = model.dlogb
    end
    return( apply_zv( stored, stepsize ) )
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

function apply_zv( stored::Storage, stepsize::Float64 )
    println("Applying ZVs to U")
    U_post = postprocess( stored.U, stored.dlogU, 1e-2 )
    println("Applying ZVs to V")
    V_post = postprocess( stored.V, stored.dlogV, 1e-2 )
    println("Applying ZVs to a")
    a_post = postprocess( stored.a, stored.dloga, 1e-2 )
    println("Applying ZVs to b")
    b_post = postprocess( stored.b, stored.dlogb, 1e-2 )
    
    println("Calculating RMSE values")
    n_iters = size( U_post, 1 )
    old_model = matrix_factorisation( train, test )
    new_model = matrix_factorisation( train, test )
    rmse_old = zeros( n_iters ) 
    rmse_new = zeros( n_iters ) 
    for i in 1:n_iters
        if i % 100 == 0
            print("$i ")
        end
        new_model.U = U_post[i,:,:]
        new_model.V = V_post[i,:,:]
        new_model.a = a_post[i,:]
        new_model.b = b_post[i,:]
        rmse_new[i] = rmse( new_model )
        old_model.U = stored.U[i,:,:]
        old_model.V = stored.V[i,:,:]
        old_model.a = stored.a[i,:]
        old_model.b = stored.b[i,:]
        rmse_old[i] = rmse( old_model )
    end
    println()
    println( "Mean RMSE:" )
    println( mean( rmse_old ) )
    println( mean( rmse_new ) )
    println( "Var RMSE:" )
    println( cov( rmse_old ) )
    println( cov( rmse_new ) )
    return( rmse_old, rmse_new )
end


function save_stored( stored::Storage, stepsize::Float64 )
    mkpath("zv")
    save( "zv/storage-$stepsize.jld", "storage", stored )
end


function postprocess( U::Array{ Float64, 3 }, dlogU::Array{ Float64, 3 }, jitter::Float64 )
    ( n_iters, D, M ) = size( U )
    # Initilise storage
    U_post = zeros( size( U ) )
    current_cov = zeros( D )
    a_current = zeros( D )
    for m in 1:M
        if m % 100 == 0
            print( "$m " )
        end
        U_curr = U[:,:,m]
        pe_curr = - 0.5 * dlogU[:,:,m]
        lw = skov.LedoitWolf()
        lw[:fit]( pe_curr )
        var_pe_inv = lw[:get_precision]()
#        var_pe = cov( pe_curr )
#        println( var_pe )
#        var_pe += jitter * mean( diag( var_pe ) ) * eye( size( var_pe, 1 ) )
#        try
#            inv( var_pe )
#        catch
#            U_post[:,:,m] = U[:,:,m]
#            continue
#        end
#        var_pe_inv = inv( var_pe )
        for d in 1:D
            current_cov = cov( vec( U_curr[:,d] ), pe_curr )
            a_current = - var_pe_inv * vec( current_cov )
            for i in 1:n_iters
                U_post[i,d,m] = U[i,d,m] + dot( a_current, vec( pe_curr[i,:] ) )
            end
        end
    end
    println()
    return( U_post )
end


function postprocess( b::Array{ Float64, 2 }, dlogb::Array{ Float64, 2 }, jitter::Float64 )
    ( n_iters, M ) = size( b )
    # Initilise storage
    b_post = zeros( size( b ) )
    current_cov = 0.0
    b_current = 0.0
    for m in 1:M
        if m % 100 == 0
            print( "$m " )
        end
        b_curr = b[:,m]
        pe_curr = - 0.5 * dlogb[:,m]
        var_pe = cov( pe_curr )
        var_pe_inv = 1 / var_pe
        if isinf( var_pe_inv )
            b_post[:,m] = b[:,m]
            continue
        end
        current_cov = cov( b_curr, pe_curr )
        a_current = - var_pe_inv * current_cov
        for i in 1:n_iters
            b_post[i,m] = b[i,m] + a_current * pe_curr[i]
        end
    end
    println()
    return( b_post )
end
