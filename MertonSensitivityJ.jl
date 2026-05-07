using Pkg; Pkg.activate("Dawei"); using Revise
using Random, Distributions, ForwardDiff

struct MertonModel{T,U,V,W,X,Y}
    S::T
    T::U
    r::V
    σ::W
    λ::X 
    δ::Y
end

abstract type OptionContract end
struct CallOption{T<:Real} <: OptionContract
    strike::T 
end

(option::CallOption)(S::Real) = max(S - option.strike, 0)
(option::CallOption)(S::AbstractVector) = @. max(S - option.strike, 0)

function bs_price(call::CallOption, S, T, r, σ)
    d1 = (log(S / call.strike) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    call_price = S * cdf(Normal(), d1) - call.strike * exp(-r * T) * cdf(Normal(), d2)
    return call_price
end

poisson_pdf(λ, n) = exp(-λ) * λ^n / factorial(n)

function price(model::MertonModel, option::OptionContract; tol = 1e-6)
    sum_so_far = eltype(model.S*model.T*model.r*model.σ*model.λ*model.δ)(0.)
    n = 0
    while true
        increment = poisson_pdf(model.λ, n) * bs_price(option, model.S, model.T, model.r, sqrt(model.σ^2 + n * model.δ^2 / model.T)) # Mean-zero jumps are simply a volatility increase, so the price is just the Black-Scholes price with an adjusted volatility; no fancy compensation terms needed. 
        sum_so_far += increment
        isnan(increment) && error("NaN encountered in price calculation at n = $n. Consider checking the parameters.")
        if abs(increment) < tol
            break
        end
        n += 1
    end
    return sum_so_far
end

radon_nikodym(lambda0, lambda, N) = exp(lambda0 - lambda) * (lambda / lambda0)^N

function test_RN(nsim::Int) # Check that the Radon-Nikodym derivative is correctly implemented by verifying that its mean is close to 1 under the original measure for various λ values. 
    Random.seed!(42) # for reproducibility
    λ₀ = 0.5
    N = rand.(Poisson(λ₀), nsim)
    rn_values = Vector{Float64}(undef, nsim)
    println("Testing Radon-Nikodym derivative implementation:")
    for λ in 0.0:0.1:1.0
        rn_values .= radon_nikodym.(λ₀, λ, N)
        println("λ: $λ, RN estimate: $(mean(rn_values)), ($(std(rn_values)/sqrt(nsim)))")
    end
end

function sim_price(model::MertonModel, option::OptionContract, λ₀::Real, nsim::Int)
    Random.seed!(42) # for reproducibility
    eps = randn(nsim)
    N = rand.(Poisson(λ₀ * model.T), nsim)
    payoffs = Vector{eltype(model.S*model.T*model.r*model.σ*model.λ*model.δ)}(undef, nsim)
    for i in eachindex(payoffs) # Can be partially (and even fully) vectorized, can you figure out how? Hint: what is the product of N independent lognormal random variables?
        if N[i] == 0
            payoffs[i] = exp(- model.r * model.T) * radon_nikodym(λ₀ * model.T, model.λ * model.T, N[i]) * option(model.S * exp((model.r - 0.5 * model.σ^2) * model.T + model.σ * sqrt(model.T) * eps[i]))
        else
            sig2 = log(1+model.δ^2)
            Yi = rand(LogNormal(-sig2/2, sqrt(sig2)), N[i])
            payoffs[i] = exp(- model.r * model.T) * radon_nikodym(λ₀ * model.T, model.λ * model.T, N[i]) * option(model.S * exp((model.r - 0.5 * model.σ^2) * model.T + model.σ * sqrt(model.T) * eps[i]) * prod(Yi))
        end
    end
    return mean(payoffs), std(payoffs) / sqrt(nsim)
end

function exact_price_and_sensitivity()
    option = CallOption(100.0)
    pr(λ) = price(MertonModel(100.0, 1.0, 0.05, 0.2, λ, 0.1), option)
    println("Exact price and sensitivity for various λ values:")
    for λ in 0.0:0.1:1.0
        println("λ: $λ, Price: $(pr(λ)), Sensitivity: $(ForwardDiff.derivative(pr, λ))")
    end
end

function simulated_price_and_sensitivity(nsim::Int)
    option = CallOption(100.0)

    pr(λ) = sim_price(MertonModel(100.0, 1.0, 0.05, 0.2, λ, 0.1), option, 0.5, nsim)
    println("Simulated price and sensitivity for various λ values, using $nsim simulations:")
    for λ in 0.0:0.1:1.0
        m, se = pr(λ)
        println("λ: $λ, Price: $(m) ($(se)), Sensitivity: $(ForwardDiff.derivative(x -> pr(x)[1], λ))")
    end
end

test_RN(1_000_000)
exact_price_and_sensitivity()
simulated_price_and_sensitivity(1_000_000)