using Pkg; Pkg.activate("Dawei"); using Revise
using Distributions, LinearAlgebra, Random, ForwardDiff

struct MertonModel{T,U,V,W,X,Y}
    S0::T
    T::U
    μ::V
    σ::W
    λ::X
    Δ::Y
end

abstract type OptionContract end
struct CallOption{T<:Real} <: OptionContract
    strike::T 
end

(option::CallOption)(S::Real) = max(S - option.strike, 0)
(option::CallOption)(S::AbstractVector) = @. max(S - option.strike, 0)

radon_nikodym(λ₀, λ, N) = exp(λ₀ - λ) .* (λ / λ₀) .^N

abstract type VibratoBasis end
struct PolyBasis <: VibratoBasis
    degree::Int
end
function (basis::PolyBasis)(x::AbstractVector, y::AbstractVector)
    m_out = Matrix{eltype(x[1]*y[1])}(undef, length(x), basis_size(basis))
    c = 0
    for i in 0:basis.degree
        for j in 0:basis.degree
            i + j ≤ basis.degree || continue
            c += 1
            m_out[:, c] .= x .^ i .* y .^ j
        end
    end
    return m_out
end
basis_size(basis::PolyBasis) = sum(1 for i in 0:basis.degree for j in 0:basis.degree if i + j ≤ basis.degree)

poisson_rand(λ::Real, N::Int) = rand(Poisson(λ), N)
poisson_rand(λ::AbstractVector, ::Int) = rand.(Poisson.(λ))


poisson_pdf(λ, n) = exp(-λ) * λ^n / factorial(n)

function price(model::MertonModel, option::OptionContract; tol = 1e-6)
    sum_so_far = eltype(model.S0*model.T*model.μ*model.σ*model.λ*0.1)(0.)
    n = 0
    while true
        increment = poisson_pdf(model.λ, n) * bs_price(option, model.S0, model.T, model.μ, sqrt(model.σ^2 + n * 0.1^2 / model.T)) # Mean-zero jumps are simply a volatility increase, so the price is just the Black-Scholes price with an adjusted volatility; no fancy compensation terms needed. 
        sum_so_far += increment
        isnan(increment) && error("NaN encountered in price calculation at n = $n. Consider checking the parameters.")
        if abs(increment) < tol
            break
        end
        n += 1
    end
    return sum_so_far
end




function sim_T(model::MertonModel, λ₀::Union{Real, AbstractVector}, nsims::Int, S₀, T::Real)
    S = Vector{Float64}(undef, nsims) # price at time T - Δ
    S .= S₀ .* exp.((model.μ - 0.5*model.σ ^ 2) .* T .+ model.σ .* sqrt(T) .* randn.()) # price at time T - Δ
    Nvec = poisson_rand(λ₀ .* T, nsims) # Poisson variables for the number of jumps
    for i in eachindex(S)
        S[i] *= rand(LogNormal(-0.5 * log(1+0.1^2) * Nvec[i], sqrt(log(1+0.1^2)) * sqrt(Nvec[i]))) # price at time T - Δ with jumps. Note that an empty product is 1, so this works even if Nvec[i] == 0. However, if Ydist is lognormal, what can you say about the product of independent lognormals? 
    end
    return S, Nvec
end

function sim_T2(model::MertonModel, λ₀::Union{Real, AbstractVector}, nsims::Int, S₀, T::Real)
    S = Vector{Float64}(undef, nsims) # price at time T - Δ
    S .= S₀ .* exp.((model.μ - 0.5*model.σ ^ 2) .* T .- model.σ .* sqrt(T) .* randn.()) # price at time T - Δ
    Nvec = poisson_rand(λ₀ .* T, nsims) # Poisson variables for the number of jumps
    for i in eachindex(S)
        S[i] *= rand(LogNormal(-0.5 * log(1+0.1^2) * Nvec[i], sqrt(log(1+0.1^2)) * sqrt(Nvec[i]))) # price at time T - Δ with jumps. Note that an empty product is 1, so this works even if Nvec[i] == 0. However, if Ydist is lognormal, what can you say about the product of independent lognormals? 
    end
    return S, Nvec
end

function sim_T_and_T_Δ(model::MertonModel, λ₀::Real, nsims::Int)
    S_T_Δ, _ = sim_T(model, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    λs = rand(Uniform(0, 2*λ₀), nsims) # Perturbed λ values for the Vibrato method, which determine the distribution of jumps on [T-Δ, T]
    S_T, _ = sim_T(model, λs, nsims, S_T_Δ, model.Δ) # Simulate prices at time T
    return S_T_Δ, S_T, λs
end

function sim_T_and_T_Δ2(model::MertonModel, λ₀::Real, nsims::Int)
    S_T_Δ, _ = sim_T2(model, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    λs = rand(Uniform(0, 2*λ₀), nsims) # Perturbed λ values for the Vibrato method, which determine the distribution of jumps on [T-Δ, T]
    S_T, _ = sim_T2(model, λs, nsims, S_T_Δ, model.Δ) # Simulate prices at time T
    return S_T_Δ, S_T, λs
end



function estimate_vibrato(model::MertonModel, option::OptionContract, λ₀::Real, nsims::Int, basis::VibratoBasis)
    Random.seed!(42) # for reproducibility
    
    S_T_Δ1, S_T1, λs = sim_T_and_T_Δ(model, λ₀, nsims)
    S_T_Δ2, S_T2, λs = sim_T_and_T_Δ2(model, λ₀, nsims)
    X1 = basis(S_T_Δ1 ./ model.S0, λs) # design matrix. Some normalization to avoid overflow for high-degree polynomials
    y1 = exp(-model.μ * model.T) * option(S_T1)
    X2 = basis(S_T_Δ2 ./ model.S0, λs) # design matrix. Some normalization to avoid overflow for high-degree polynomials
    y2 = exp(-model.μ * model.T) * option(S_T2)
    return X1 \ y1, X2 \ y2 # Return the regression coefficients
end

function price_vibrato(model::MertonModel, λ₀::Real, nsims::Int, basis::VibratoBasis, β̂1::AbstractVector, β̂2::AbstractVector)
    Random.seed!(43) # for reproducibility
    S_T_Δ1, Nvec1 = sim_T(model, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    λ_vec = fill(model.λ, nsims) # Use the original λ₀ for pricing
    ŷ1 = basis(S_T_Δ1 ./ model.S0, λ_vec) * β̂1 # Predicted payoffs at time T - Δ using the regression coefficients
    S_T_Δ2, Nvec2 = sim_T2(model, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    ŷ2 = basis(S_T_Δ2 ./ model.S0, λ_vec) * β̂2 # Predicted payoffs at time T - Δ using the regression coefficients
    y_after_RN1 = ŷ1 .* radon_nikodym(λ₀ * (model.T - model.Δ), model.λ * (model.T - model.Δ), Nvec1) # Adjust the predicted payoffs using the Radon-Nikodym derivative to account for the change in measure
    y_after_RN2 = ŷ2 .* radon_nikodym(λ₀ * (model.T - model.Δ), model.λ * (model.T - model.Δ), Nvec2) # Adjust the predicted payoffs using the Radon-Nikodym derivative to account for the change in measure
    return mean(y_after_RN1), std(y_after_RN1) / sqrt(nsims), mean(y_after_RN2), std(y_after_RN2) / sqrt(nsims) # Return the estimated price and its standard error
end

function simulated_price_and_sensitivity_vibrato(nsim::Int, nsim_estimate::Int)
    option = CallOption(100.0)
    basis = PolyBasis(5)
    model(λ) = MertonModel(100.0, 1.0, 0.05, 0.2, λ, 1e-8)
    # jumps = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2))
    λ₀ = 0.5
    
    println("Simulated price and sensitivity for various λ values, using $nsim simulations:")
    for λ in 0.0:0.1:1.0
        mod = model(λ)
        β̂1, β̂2 = estimate_vibrato(mod, option, λ₀, nsim_estimate, basis) # removed jumps
        m1, se1, m2, se2 = price_vibrato(mod, λ₀, nsim, basis, β̂1, β̂2)
        m = (m1 + m2) / 2
        se = sqrt(se1^2 + se2^2) / 2
        exact = price(mod, option)
        println("λ: $λ, Price: $(m) ($(se)), MSE = $(abs(m .- exact).^2))")
        # S_final, rn = sim_T(mod, λ₀, nsim, mod.S0, mod.T)
        # S_rn_disc = exp.(-mod.μ * mod.T) .* option(S_final) .* radon_nikodym(λ₀ * mod.T, mod.λ * mod.T, rn) # Just a check for reference
        # println("- Direct price: $(mean(S_rn_disc)) ($(std(S_rn_disc) / sqrt(nsim)))")
    end
end
    
simulated_price_and_sensitivity_vibrato(100_000, 1000_000)