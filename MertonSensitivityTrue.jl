using Distributions, ForwardDiff, Plots
function BlackScholes(S, K, T, r, σ)
    d1 = (log(S / K) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    call_price = S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
    return call_price
end

function SumTerms(S, K, T, r, sigma, lambda, kappa, delta, tol = 1e-6)
    old = 0.0
    new = 0.0
    n = 0
    gamma = log(1+kappa)
    lambdaprime = lambda *(1+kappa)
    function Poisson(lambda, n)
        return exp(-lambda) * lambda^n / factorial(n)
    end
    while true
        new = old + ForwardDiff.derivative(lambda -> Poisson(lambdaprime, n) * BlackScholes(S, K, T, r - lambda * kappa + n * gamma /T, sqrt(sigma^2 + n * delta^2 / T)), lambda)
        if abs(new - old) < tol
            break
        end
        old = new
        n += 1
    end
    return new
end

function greek_sensitivity(lambda)
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    kappa = 0.1
    delta = 0.1
    return SumTerms(S, K, T, r, sigma, lambda, kappa, delta)
end

lambda_values = 0.0:0.1:1
sensitivities_true = [greek_sensitivity(lambda) for lambda in lambda_values]
println("The sensitivities, under the true model, are: ", sensitivities_true)

