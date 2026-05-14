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
struct LaguerreBasis <: VibratoBasis
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

function (basisL::LaguerreBasis)(x::AbstractVector, y::AbstractVector)
    m_out = Matrix{eltype(x[1]*y[1])}(undef, length(x), basis_size(basisL))
    c = 0
    for i in 0:basisL.degree
        for j in 0:basisL.degree
            i + j ≤ basisL.degree || continue
            c += 1
            m_out[:, c] .= laguerre.(i, x) .* laguerre.(j, y)
        end
    end
    return m_out
end
basis_size(basisL::LaguerreBasis) = sum(1 for i in 0:basisL.degree for j in 0:basisL.degree if i + j ≤ basisL.degree)



poisson_rand(λ::Real, N::Int) = rand(Poisson(λ), N)
poisson_rand(λ::AbstractVector, ::Int) = rand.(Poisson.(λ))

function sim_T(model::MertonModel, λ₀::Union{Real, AbstractVector}, nsims::Int, S₀, T::Real)
    S = Vector{Float64}(undef, nsims) # price at time T - Δ
    S .= S₀ .* exp.((model.μ - 0.5*model.σ ^ 2) .* T .+ model.σ .* sqrt(T) .* randn.()) # price at time T - Δ
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

function estimate_vibrato(model::MertonModel, option::OptionContract, λ₀::Real, nsims::Int, basis::VibratoBasis)
    Random.seed!(42) # for reproducibility
    
    S_T_Δ, S_T, λs = sim_T_and_T_Δ(model, λ₀, nsims)

    X = basis(S_T_Δ ./ model.S0, λs) # design matrix. Some normalization to avoid overflow for high-degree polynomials
    y = exp(-model.μ * model.T) * option(S_T)
    return X\y # Return the regression coefficients
end

function price_vibrato(model::MertonModel, λ₀::Real, nsims::Int, basis::VibratoBasis, β̂::AbstractVector)
    Random.seed!(43) # for reproducibility
    S_T_Δ, Nvec = sim_T(model, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    λ_vec = fill(model.λ, nsims) # Use the original λ₀ for pricing
    ŷ = basis(S_T_Δ ./ model.S0, λ_vec) * β̂ # Predicted payoffs at time T - Δ using the regression coefficients

    y_after_RN = ŷ .* radon_nikodym(λ₀ * (model.T - model.Δ), model.λ * (model.T - model.Δ), Nvec) # Adjust the predicted payoffs using the Radon-Nikodym derivative to account for the change in measure
    return mean(y_after_RN), std(y_after_RN) / sqrt(nsims) # Return the estimated price and its standard error
end

function simulated_price_and_sensitivity_vibrato(nsim::Int, nsim_estimate::Int)
    option = CallOption(100.0)
    basis = PolyBasis(5)
    model(λ) = MertonModel(100.0, 1.0, 0.05, 0.2, λ, 1/30)
    # jumps = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2))
    λ₀ = 0.5
    
    println("Simulated price and sensitivity for various λ values, using $nsim simulations:")
    for λ in 0.0:0.1:1.0
        mod = model(λ)
        β̂ = estimate_vibrato(mod, option, λ₀, nsim_estimate, basis) # removed jumps
        m, se = price_vibrato(mod, λ₀, nsim, basis, β̂)
        println("λ: $λ, Price: $(m) ($(se)), Sensitivity: $(ForwardDiff.derivative(x -> price_vibrato(model(x), λ₀, nsim, basis, β̂)[1], λ)), Second-order: $(ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> price_vibrato(model(y), λ₀, nsim, basis, β̂)[1], x), λ))")
        S_final, rn = sim_T(mod, λ₀, nsim, mod.S0, mod.T)
        S_rn_disc = exp.(-mod.μ * mod.T) .* option(S_final) .* radon_nikodym(λ₀ * mod.T, mod.λ * mod.T, rn) # Just a check for reference
        println("- Direct price: $(mean(S_rn_disc)) ($(std(S_rn_disc) / sqrt(nsim)))")
    end
end
    
simulated_price_and_sensitivity_vibrato(100_000, 1000_000)