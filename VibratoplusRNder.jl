using Pkg; Pkg.activate("DaweiVibrato"); using Revise
using Distributions, Plots, LinearAlgebra, Random

struct MertonModel{T,U,V,W,X,Z}
    S0::T
    T::U
    μ::V
    σ::W
    λ::X 
    dt::Z
end


abstract type OptionContract end
struct CallOption{T<:Real} <: OptionContract
    strike::T 
end

(option::CallOption)(S::Real) = max(S - option.strike, 0)
(option::CallOption)(S::AbstractVector) = @. max(S - option.strike, 0)


function MJD_FixedJumps(model::MertonModel, Ydist)
    timepoints = 0:model.dt:model.T
    X0 = log(model.S0)
    stockprice = vcat([X0], Vector{eltype(X0)}(undef, length(timepoints)-1))
    N = rand(Poisson(model.λ*model.dt),length(timepoints)-1) 
    for i in 1:length(timepoints)-1 
        Z = randn()
        if N[i] == 0
            M = 0
        else
            M = sum(rand(Ydist, N[i]))
        end
        stockprice[i+1] = stockprice[i] + (model.μ - 0.5*model.σ ^ 2) * model.dt + model.σ * sqrt(model.dt) * Z + M
    end
    return exp.(stockprice), N
end

function Vibrato(model::MertonModel, option::OptionContract, Ydist, nsims::Int, p::Int)
    Random.seed!(42) # for reproducibility
    Δ= model.dt
    θ = rand(nsims)
    X = Matrix{Float64}(undef,nsims, p + 2) # design matrix
    y = Vector{Float64}(undef, nsims) # payoffs
    for i in eachindex(θ)
        stocks = MJD_FixedJumps(MertonModel(model.S0, model.T, model.μ, model.σ, θ[i], Δ), Ydist)[1]
        S_T = exp(-model.μ * model.T) * option(stocks[end])
        S_T_Δ = stocks[end - 1]
        X[i,:] = vcat([S_T_Δ ^ k for k in 0:p], θ[i])
        y[i] = S_T
    end
    return X, y
end


radon_nikodym(lambda0, lambda, N) = exp(lambda0 - lambda) * (lambda / lambda0) .^N


function sim_price_vibrato(model::MertonModel, option::OptionContract, λ₀::Real, nsim::Int)
    Random.seed!(42) # for reproducibility
    Ydist = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2))
    X,y = Vibrato(model, option, Ydist, nsim, 2)
    bhat = X\y
    yfit = X * bhat
    return mean(yfit .* radon_nikodym.(λ₀, model.λ,MJD_FixedJumps(model, Ydist)[2]))
end

# sim_price_vibrato(MertonModel(100,1,0.05,0.2,0.5,1/252), CallOption(100), 0.5, 10000)





function simprice(nsim::Int64)
    Random.seed!(42) # for reproducibility
    price = Vector{Float64}(undef, nsim)
    for i in 1:nsim
        price[i] = MJD_FixedJumps(MertonModel(100, 1, 0.05, 0.2, 0.5, 1/252), LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2)))[1][end]
    end
    return price
end
simprice(10000)

Z = randn(10000)
Ydist = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2))
N = rand.(Poisson(0.5 * 1/252), 10000)
100 .* exp.(0.05 * 1 + (-0.5 * 0.2^2) * 1 + 0.2 * sqrt(1/252) .* Z)
# test whether the two simulations are identical (simulating the entire path vs simulating just the terminal price using the exact distribution of the terminal price) by comparing their means and variances.