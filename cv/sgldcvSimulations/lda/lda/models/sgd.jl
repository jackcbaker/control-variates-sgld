include("lda.jl")

type sgd
    subsize::Int64
    iter::Int64
    subsample::Array{Int64,1}
    ϵ::Float64
    G::Array{Float64,2}
    ll::Array{Float64,2}
    theta::Array{Float64,2}
end

function sgd( model::lda, subsize, ϵ )
    subsample = sample( 1:model.M, subsize )
    G = ones( size( model.theta ) )
    ll = zeros( size( model.theta ) )
    theta = model.theta
    sgd( subsize, 1, subsample, ϵ, G, ll, theta )
end

"""
Update one step of stochastic gradient descent
"""
function sgd_update( model::lda, tuning::sgd )
    tuning.subsample = sample( 1:model.M, tuning.subsize )
    γ = 1 / ( 1 + tuning.iter )
    ζ = 0.1
    model.zcounts[tuning.subsample,:,:] = update_topics( model, tuning.subsample )
    dlogθ = dlogpost( model, tuning.subsample )
#    tuning.G += ζ*( dlogθ.^2 - tuning.G )
    model.theta += tuning.ϵ / 2 * dlogθ
    tuning.theta = model.theta
end
