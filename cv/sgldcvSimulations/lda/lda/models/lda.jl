using StatsBase
using Distributions

"""
Latent dirichlet allocation object
"""
type lda
    name::AbstractString
    M::Int64                        # Number of documents
    V::Int64                        # Number of words
    K:: Int64                       # Number of topics
    theta::Array{ Float64, 2 }      # Array of topic-word reparameterisation
    train::Array{ Int64, 2 }        # Training set: document term matrix
    test::Array{ Int64, 2 }         # Test set: document term matrix
    zcounts::Array{ Int64, 3 }    # Latent topic assignment counts
    beta::Float64                   # Hyperparameter for phi
    alpha::Float64                  # Hyperparameter for document-topics
end

"""
Create lda object from data, with uninformative priors and start values simulated from prior.
"""
function lda( train::Array{ Int64, 2 }, test::Array{ Int64, 2 }, K::Int )
    ( M, V ) = size( train )
    beta = 1
    alpha = 1
    theta = log( rand( Gamma( beta[1], 1 ), ( K, V ) ) )
    zcounts = prior_zcounts( train, K )

    return( lda( "lda", M, V, K, theta, train, test, zcounts, beta, alpha) )
end

"""
Create initial zcounts object
"""
function prior_zcounts( train, K )
    ( M, V ) = size( train )
    zcounts = zeros( Int64, ( M, K, V ) )
    for m in 1:M
        for v in 1:V
            if train[m,v] == 0
                continue
            end
            zcounts[m,:,v] = rand( Multinomial( train[m,v], K ), 1 )
        end
    end
    return zcounts
end

"""
Calculate gradient of log posterior at current state
"""
function dlogpost( model::lda, subsample )
    subsize = size( subsample, 1 )
    dtheta = zeros( model.K, model.V )
    dtheta += grad_theta_prior( model, model.theta )
    dtheta += model.M / subsize * grad_theta( model, subsample, model.theta )
    return( dtheta )
end

"""
Calculate gradient of log posterior at the mode
"""
function pe( model::lda, subsample, theta )
    subsize = size( subsample, 1 )
    dtheta = zeros( model.K, model.V )
    dtheta += grad_theta_prior( model, theta )
    dtheta += model.M / subsize * grad_theta( model, subsample, theta )
    return( dtheta )
end

"""
Gradient of prior for theta
"""
function grad_theta_prior( model::lda, theta )
    dtheta = zeros( model.K, model.V )
    for k in 1:model.K
        for v in 1:model.V
            dtheta[k,v] += model.beta - exp( theta[k,v] )
        end
    end
    return dtheta
end

"""
Noisy gradient of lda model wrt theta
"""
function grad_theta( model::lda, m_subsample::Array{ Int64, 1 }, theta )
    dtheta = zeros( model.K, model.V )
    exp_theta = exp( theta )
    for k in 1:model.K
        for m in m_subsample
            zcount_theta_ratio = sum( model.zcounts[m,k,:] )/sum( exp_theta[k,:] )
            for v in 1:model.V
                dtheta[k,v] += model.zcounts[m,k,v] - exp_theta[k,v]*zcount_theta_ratio
            end
        end
    end
    return( dtheta )
end

"""
Update topic assignments
"""
function update_topics( model::lda, m_subsample::Array{ Int64, 1 } )
    zcounts_update = zeros( Int64, length( m_subsample ), model.K, model.V )
    exp_theta = exp( model.theta )
    ndz = sum( model.zcounts, 3 )
    prob_current = zeros( model.K )
    for i in 1:length( m_subsample )
        m = m_subsample[i]
        zcounts_current = findnz( model.zcounts[m,:,:] )
        num_nonzeros = length( zcounts_current[1] )
        for j in 1:num_nonzeros
            k_current = zcounts_current[1][j]
            v = zcounts_current[2][j]
            n_words = zcounts_current[3][j]
            denom = 0
            for k in 1:model.K
                if( k == k_current )
                    prob_current[k] = ( model.alpha + ndz[m,k] - 1 )*exp( model.theta[k,v] )
                    denom += prob_current[k]
                else
                    prob_current[k] = ( model.alpha + ndz[m,k] )*exp( model.theta[k,v] )
                    denom += prob_current[k]
                end
            end
            prob_current ./= denom
            zcounts_update[i,:,v] += rand( Multinomial( n_words, prob_current ) )
        end
    end
    return zcounts_update
end

"""
Calculate perplexity given test dataset
"""
function perplexity( model::lda, test::Array{ Int64, 2 }, avg_theta::Array{ Float64, 2 } )
    perplexity = 0
    word_topic_est = exp( avg_theta ) ./ sum( exp( avg_theta ), 2 )
    doc_topic_est = float( sum( model.zcounts[:,:,:], 3 ) + 1 ) 
    for m in 1:model.M
        doc_topic_est[m,:] /= ( sum( findnz( model.zcounts[m,:,:] )[3] ) + model.K )
    end
    for m in 1:model.M
        for v in 1:model.V
            # Save computation by skipping zero words
            if ( test[m,v] == 0 )
                continue
            end
            perplex_current = 0
            for k in 1:model.K
                perplex_current += sum( doc_topic_est[m,k,:] )*word_topic_est[k,v]
            end
            perplexity += test[m,v]*log( perplex_current )
        end
    end
    perplexity = exp( -perplexity/sum( test ) )
    return perplexity
end

"""
Calculate perplexity given test dataset
"""
function perplexity( model::lda, test::Array{ Int64, 2 } )
    perplexity = 0
    word_topic_est = exp( model.theta ) ./ sum( exp( model.theta ), 2 )
    doc_topic_est = float( sum( model.zcounts[:,:,:], 3 ) + 1 ) 
    for m in 1:model.M
        doc_topic_est[m,:] /= ( sum( findnz( model.zcounts[m,:,:] )[3] ) + model.K )
    end
    for m in 1:model.M
        for v in 1:model.V
            # Save computation by skipping zero words
            if ( test[m,v] == 0 )
                continue
            end
            perplex_current = 0
            for k in 1:model.K
                perplex_current += sum( doc_topic_est[m,k,:] )*word_topic_est[k,v]
            end
            perplexity += test[m,v]*log( perplex_current )
        end
    end
    perplexity = exp( -perplexity/sum( test ) )
    return perplexity
end

"""
Calculate likelihood of the model
"""
function likelihood( model::lda )
    llik = 0
    for k in 1:model.K
        denom = log( sum( exp( model.theta[k,:] ) ) )
        for m in 1:model.M
            for v in 1:model.V
                llik += model.zcounts[m,k,v]*( model.theta[k,v] - denom )
            end
        end
    end
    return llik
end
