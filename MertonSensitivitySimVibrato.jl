# Jumps are Poisson at specific timepoints

using Distributions, Plots, LinearAlgebra, Random
function MJD_FixedJumps(S0, mu, T, sigma, lambda, dt, Ydist)
    timepoints = 0:dt:T
    X0 = log(S0)
    stockprice = vcat([X0], zeros(length(timepoints)-1))
    for i in 1:length(timepoints)-1
        Z = randn()
        N = rand(Poisson(lambda*dt))
        if N == 0
            M = 0
        else
            M = sum(rand(Ydist, N))
        end
        stockprice[i+1] = stockprice[i] + (mu - 0.5*sigma ^ 2) * dt + sigma * sqrt(dt) * Z + M
    end
    return exp.(stockprice)
end

# Simulating Jumps times

function MJD_RandomJumps(S0, mu, sigma, T, lambda, jumpfreq, Ydist)
    tau0 = 0
    X0 = log(S0)
    jumptimes = vcat([tau0], zeros(jumpfreq))
    for i in 1:jumpfreq
        R = rand(Exponential(1/lambda))
        jumptimes[i+1] = jumptimes[i] + R
    end
    jumptimes = jumptimes[jumptimes .<= T]
    stockprice = vcat([X0], zeros(length(jumptimes)-1))
    for i in eachindex(stockprice)[2:end]
        Z = randn()
        logY = log(rand(Ydist))
        stockprice[i] = stockprice[i-1] + (mu - 0.5*sigma ^ 2) * (jumptimes[i] - jumptimes[i-1]) + sigma * sqrt(jumptimes[i] - jumptimes[i-1]) * Z + logY
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
        stockprice = MJD_FixedJumps(S0, mu, T, sigma, lambda, dt, Ydist)
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
    StockBefore[i] = MJD_FixedJumps(100, 0.05, 1.0, 0.2, 0.1, 0.01, LogNormal(mu, sig))[end-1]
    OptionsPayoff[i] = exp(-0.05 * 1.0) * eu_call_payoff(MJD_FixedJumps(100, 0.05, 1.0, 0.2, 0.1, 0.01, LogNormal(mu, sig))[end], 100)
end

theta = LinRange(0, 1, length(StockBefore))
X = hcat([StockBefore .^ i for i in 0:3]..., theta)
y = OptionsPayoff

bhat = X \ y


# simulate_option_payoff(10000, "digital_call", 100, 100, 0.05, 1.0, 0.2, 0.1, 0.01, LogNormal(mu, sig))