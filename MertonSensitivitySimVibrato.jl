# Jumps are Poisson at specific timepoints
using Distributions, Plots, LinearAlgebra, Random


struct MertonModel{T,U,V,W,X,Y,Z}
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
    stockprice = vcat([X0], zeros(length(timepoints)-1))
    for i in 1:length(timepoints)-1 
        Z = randn()
        N = rand(Poisson(model.λ*model.dt)) 
        if N == 0
            M = 0
        else
            M = sum(rand(Ydist, N))
        end
        stockprice[i+1] = stockprice[i] + (model.μ - 0.5*model.σ ^ 2) * model.dt + model.σ * sqrt(model.dt) * Z + M
    end
    return exp.(stockprice)
end

# Simulating Jumps times

function MJD_RandomJumps(model::MertonModel, Ydist)
    tau0 = 0
    X0 = log(model.S0)
    jumptimes = vcat([tau0], zeros(model.λ))
    for i in 1:model.λ
        R = rand(Exponential(1/model.λ))
        jumptimes[i+1] = jumptimes[i] + R
    end
    jumptimes = jumptimes[jumptimes .<= model.T]
    stockprice = vcat([X0], zeros(length(jumptimes)-1))
    for i in eachindex(stockprice)[2:end]
        Z = randn()
        logY = log(rand(Ydist))
        stockprice[i] = stockprice[i-1] + (model.μ - 0.5*model.σ ^ 2) * (jumptimes[i] - jumptimes[i-1]) + model.σ * sqrt(jumptimes[i] - jumptimes[i-1]) * Z + logY
    end
    return exp.(stockprice)
end


function eu_call_payoff(S, K)
    return max(S - K, 0)
end

function eu_put_payoff(S, K)
    return max(K - S, 0)
end

function digital_call_payoff(S, K)
    return Float64(S>K)
end

function simulate_option_payoff(nsims, style, S0, K, mu, T, sigma, lambda, dt, Ydist)
    payoffs = zeros(nsims)
    for i in eachindex(payoffs)
        stockprice = MJD_FixedJumps(MertonModel(S0, T, mu, sigma, lambda, dt), Ydist)
        if style == "call"
            payoffs[i] = eu_call_payoff(stockprice[end], K)
        elseif style == "put"
            payoffs[i] = eu_put_payoff(stockprice[end], K)
        elseif style == "digital_call"
            payoffs[i] = digital_call_payoff(stockprice[end], K)
        end
    end
    return mean(payoffs)
end

k = 0
delta = 0.1
muX = 1+k
mu = log(muX^2/(sqrt(muX^2 + delta^2)))
sig = sqrt(log(1 + delta^2/muX^2))

StockBefore = zeros(1000)
OptionsPayoff = zeros(1000)
Random.seed!(42) # for reproducibility
for i in eachindex(StockBefore)
    StockBefore[i] = MJD_FixedJumps(MertonModel(100, 0.05, 1.0, 0.2, 0.1, 0.01), LogNormal(mu, sig))[end-1]
    OptionsPayoff[i] = exp(-0.05 * 1.0) * eu_call_payoff(MJD_FixedJumps(MertonModel(100, 0.05, 1.0, 0.2, 0.1, 0.01), LogNormal(mu, sig))[end], 100)
end

theta = LinRange(0, 1, length(StockBefore))
X = hcat([StockBefore .^ i for i in 0:3]..., theta)
y = OptionsPayoff

bhat = X \ y


# simulate_option_payoff(10000, "digital_call", 100, 100, 0.05, 1.0, 0.2, 0.1, 0.01, LogNormal(mu, sig))