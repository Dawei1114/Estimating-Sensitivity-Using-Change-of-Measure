using Distributions, ForwardDiff

function RadonNikodym(lambda0,lambda,N)
    return exp(lambda0 - lambda) * (lambda/lambda0) .^ N
end

function MertonSensitivitySim(nsim,r,g, S0, sigma, lambda0, lambda, T, kappa, delta)
    eps = randn(nsim)
    N = rand.(Poisson(lambda0), nsim)
    dd_lambda = zeros(nsim)
    sumterms = zeros(nsim)
    for i in eachindex(sumterms)
        if N[i] == 0
            sumterms[i] = ForwardDiff.derivative(x -> g(S0 * exp((r - 0.5 * sigma ^ 2 - x * kappa) * T + sigma * sqrt(T) * eps[i])) *  RadonNikodym(lambda0, x, N[i]), lambda)
        else
            Yi = rand(LogNormal(0, delta), N[i])
            sumterms[i] = ForwardDiff.derivative(x -> g(S0 * exp((r - 0.5 * sigma ^ 2 - x * kappa) * T + sigma * sqrt(T) * eps[i]) * prod(Yi)) * RadonNikodym(lambda0, x, N[i]), lambda)
        end
    end
    return mean(sumterms)
end

function g(x)
    return max.(x .- 100, 0)
end

lambda_values = 0.0:0.1:1

lambda0vec = 0.1:0.0001:1
sensitivities_sim = zeros(length(lambda_values))

for i in eachindex(lambda_values)
    lambda0init = zeros(length(lambda0vec))
    for j in eachindex(lambda0init)
        lambda0init[j] = MertonSensitivitySim(10000, 0.05, g, 100, 0.2, lambda0vec[j], lambda_values[i], 1, 0.1, 0.1)
    end
    sensitivities_sim[i] = mean(lambda0init)
end

sensitivities_sim
println("The sensitivities, under the simulated model, are: ", sensitivities_sim)


