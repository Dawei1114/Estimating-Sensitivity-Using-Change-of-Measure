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


abstract type VarianceReductionTechnique end
struct AntitheticVariates <: VarianceReductionTechnique end
abstract type VibratoBasis end

struct PolyBasis <: VibratoBasis
    degree::Int
end

struct LaguerreBasis <: VibratoBasis
    degree::Int
end

struct HermiteBasis <: VibratoBasis
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

function laguerre(n, x)
    if n == 0
        return 1.0
    elseif n == 1
        return 1.0 - x
    else
        Lₙ₋₂ = 1.0
        Lₙ₋₁ = 1.0 - x
        for k in 2:n
            Lₙ = ((2k - 1 - x) * Lₙ₋₁ - (k - 1) * Lₙ₋₂) / k
            Lₙ₋₂, Lₙ₋₁ = Lₙ₋₁, Lₙ
        end
        return Lₙ₋₁
    end
end

function (basisL::LaguerreBasis)(x::AbstractVector, y::AbstractVector)
    m_out = Matrix{eltype(x[1]*y[1])}(undef, length(x), basis_sizeL(basisL))
    c = 0
    for i in 0:basisL.degree
        for j in 0:basisL.degree
            i + j ≤ basisL.degree || continue
            c += 1
            m_out[:, c] .= exp.(-x) .* laguerre.(i, x) .* laguerre.(j, y)
        end
    end
    return m_out
end
basis_sizeL(basis::LaguerreBasis) = sum(1 for i in 0:basis.degree for j in 0:basis.degree if i + j ≤ basis.degree)


function hermite(n, x)
    if n == 0
        return 1.0
    elseif n == 1
        return 2.0 * x
    else
        Hₙ₋₂ = 1.0
        Hₙ₋₁ = 2.0 * x
        for k in 2:n
            Hₙ = 2.0 * x .* Hₙ₋₁ - 2.0 * (k - 1) * Hₙ₋₂
            Hₙ₋₂, Hₙ₋₁ = Hₙ₋₁, Hₙ
        end
        return Hₙ₋₁
    end
end


function(basisH::HermiteBasis)(x::AbstractVector, y::AbstractVector)
    m_out = Matrix{eltype(x[1]*y[1])}(undef, length(x), basis_sizeH(basisH))
    c = 0
    for i in 0:basisH.degree
        for j in 0:basisH.degree
            i + j ≤ basisH.degree || continue
            c += 1
            m_out[:, c] .= exp.(-0.5 * (x.^2 + y.^2)) .* hermite.(i, x) .* hermite.(j, y)
        end
    end
    return m_out
end

basis_sizeH(basis::HermiteBasis) = sum(1 for i in 0:basis.degree for j in 0:basis.degree if i + j ≤ basis.degree)



poisson_rand(λ::Real, N::Int) = rand(Poisson(λ), N)
poisson_rand(λ::AbstractVector, ::Int) = rand.(Poisson.(λ))

function sim_T(model::MertonModel,Z::AbstractVector, λ₀::Union{Real, AbstractVector}, nsims::Int, S₀, T::Real)
    S = Vector{Float64}(undef, nsims) # price at time T - Δ
    S .= S₀ .* exp.((model.μ - 0.5*model.σ ^ 2) .* T .+ model.σ .* sqrt(T) .* Z) # price at time T - Δ
    Nvechalf = poisson_rand(λ₀ .* T, nsims ÷ 2)
    Nvec = vcat(Nvechalf, Nvechalf) # Poisson variables for the number of jumps
    for i in eachindex(S)
        S[i] *= rand(LogNormal(-0.5 * log(1+0.1^2) * Nvec[i], sqrt(log(1+0.1^2)) * sqrt(Nvec[i]))) # price at time T - Δ with jumps. Note that an empty product is 1, so this works even if Nvec[i] == 0. However, if Ydist is lognormal, what can you say about the product of independent lognormals? 
    end
    return S, Nvec
end

function sim_T_and_T_Δ(::AntitheticVariates, model::MertonModel, λ₀::Real, nsims::Int)
    Z = randn(nsims ÷ 2) # Standard normal random variables for the Brownian motion component
    Zfull = vcat(Z, -Z) # Antithetic variates for variance reduction
    S_T_Δ, _ = sim_T(model, Zfull, λ₀, nsims, model.S0, model.T - model.Δ)
    λhalf = rand(Uniform(0, 2*λ₀), nsims ÷ 2) # Perturbed λ values for the Vibrato method, which determine the distribution of jumps on [T-Δ, T]
    λs = vcat(λhalf, λhalf)
    S_T, _ = sim_T(model, Zfull, λs, nsims, S_T_Δ, model.Δ)
    return S_T_Δ, S_T, λs
end

function estimate_vibrato(vrt::AntitheticVariates, model::MertonModel, option::OptionContract, λ₀::Real, nsims::Int, basis::VibratoBasis)
    Random.seed!(42) # for reproducibility
    
    S_T_Δ, S_T, λs = sim_T_and_T_Δ(vrt, model, λ₀, nsims)

    X1= basis(S_T_Δ[1:nsims÷2] ./ model.S0, λs[1:nsims÷2]) # design matrix. Some normalization to avoid overflow for high-degree polynomials
    y1 = exp(-model.μ * model.T) * option(S_T[1:nsims÷2])

    X2 = basis(S_T_Δ[nsims÷2+1:nsims] ./ model.S0, λs[nsims÷2+1:nsims]) 
    y2 = exp(-model.μ * model.T) * option(S_T[nsims÷2+1:nsims]) #
    return X1\y1, X2\y2 # Return the regression coefficients for both halves of the antithetic sample
end

function price_vibrato(::AntitheticVariates,model::MertonModel, λ₀::Real, nsims::Int, basis::VibratoBasis, β̂::Tuple{AbstractVector, AbstractVector})
    Random.seed!(43) # for reproducibility
    Z = randn(nsims ÷ 2) # Standard normal random variables for the Brownian motion component
    Zfull = vcat(Z, -Z) # Antithetic variates for variance reduction
    S_T_Δ, Nvec = sim_T(model, Zfull, λ₀, nsims, model.S0, model.T - model.Δ) # Simulate prices at time T - Δ
    λ_vec = fill(model.λ, nsims) # Use the original λ₀ for pricing
    ŷ1 = basis(S_T_Δ[1:nsims÷2] ./ model.S0, λ_vec[1:nsims÷2]) * β̂[1]
    ŷ2 = basis(S_T_Δ[nsims÷2+1:nsims] ./ model.S0, λ_vec[nsims÷2+1:nsims]) * β̂[2]
    y_after_RN1 = ŷ1 .* radon_nikodym(λ₀ * (model.T - model.Δ), model.λ * (model.T - model.Δ), Nvec[1:nsims÷2]) # Adjust the predicted payoffs using the Radon-Nikodym derivative to account for the change in measure
    y_after_RN2 = ŷ2 .* radon_nikodym(λ₀ * (model.T - model.Δ), model.λ * (model.T - model.Δ), Nvec[nsims÷2+1:nsims]) # Adjust the predicted payoffs using the Radon-Nikodym derivative to account for the change in measure
    return mean(y_after_RN1 .+ y_after_RN2) / 2, std(y_after_RN1 .+ y_after_RN2) / (2 * sqrt(nsims)) # Return the estimated price and its standard error
end

function simulated_price_and_sensitivity_vibrato(::AntitheticVariates, nsim::Int, nsim_estimate::Int)
    option = CallOption(100.0)
    basis = LaguerreBasis(5)
    model(λ) = MertonModel(100.0, 1.0, 0.05, 0.2, λ, 1e-10)
    # jumps = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2))
    λ₀ = 0.6 # 
    
    println("Simulated price and sensitivity for various λ values, using $nsim simulations:")
    for λ in 0.0:0.1:1.0
        mod = model(λ)
        vrt = AntitheticVariates()
        β̂1 , β̂2= estimate_vibrato(vrt, mod, option, λ₀, nsim_estimate, basis) 
        m, se = price_vibrato(vrt, mod, λ₀, nsim, basis, (β̂1, β̂2))
        println("λ: $λ, Price: $(m) ($(se)), Sensitivity: $(ForwardDiff.derivative(x -> price_vibrato(vrt, model(x), λ₀, nsim, basis, (β̂1, β̂2))[1], λ)), Second-order Sensitivity: $(ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> price_vibrato(vrt, model(y), λ₀, nsim, basis, (β̂1, β̂2))[1], x), λ))")
        # S_final, rn = sim_T(mod, λ₀, nsim, mod.S0, mod.T)
        # S_rn_disc = exp.(-mod.μ * mod.T) .* option(S_final) .* radon_nikodym(λ₀ * mod.T, mod.λ * mod.T, rn) # Just a check for reference
        # println("- Direct price: $(mean(S_rn_disc)) ($(std(S_rn_disc) / sqrt(nsim)))")
    end
end

simulated_price_and_sensitivity_vibrato(AntitheticVariates(), 2 * 100_000, 2 * 1000_000)