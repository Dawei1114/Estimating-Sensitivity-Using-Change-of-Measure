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

function Vibrato(model::MertonModel, option::OptionContract, λ₀::Real, Ydist, nsims::Int, p::Int, m::Int)
    Random.seed!(42) # for reproducibility
    Δ= model.dt
    Tnum = promote_type(typeof(model.λ), typeof(λ₀))
    X = Matrix{Tnum}(undef, nsims, p + m + 1) # design matrix
    y = Vector{Tnum}(undef, nsims) # payoffs
    Nvec = Vector{Int}(undef, nsims) # Poisson variables for the Radon-Nikodym derivative
    for i in eachindex(X[:,1])
        N = rand.(Poisson(λ₀*(model.T - Δ)))
        Nvec[i] = N
        if N == 0
            S_T_Δ = model.S0 * exp((model.μ - 0.5*model.σ ^ 2) * (model.T - Δ) + model.σ * sqrt(model.T - Δ) * randn()) # price at time T - Δ
            M = rand(Poisson(λ₀*Δ))
            if M == 0
                S_T = S_T_Δ * exp((model.μ - 0.5*model.σ ^ 2) * Δ + model.σ * sqrt(Δ) * randn())
            else
                Yi = rand(Ydist, M)
                S_T = S_T_Δ * exp((model.μ - 0.5*model.σ ^ 2) * Δ + model.σ * sqrt(Δ) * randn()) * prod(Yi)
            end
        else
            Yi = rand(Ydist, N)
            S_T_Δ = model.S0 * exp((model.μ - 0.5*model.σ ^ 2) * (model.T - Δ) + model.σ * sqrt(model.T - Δ) * randn()) * prod(Yi)
            M = rand(Poisson(λ₀*Δ))
            if M == 0
                S_T = S_T_Δ * exp((model.μ - 0.5*model.σ ^ 2) * Δ + model.σ * sqrt(Δ) * randn())
            else
                Yi = rand(Ydist, M)
                S_T = S_T_Δ * exp((model.μ - 0.5*model.σ ^ 2) * Δ + model.σ * sqrt(Δ) * randn()) * prod(Yi)
            end
        end   
        X[i,:] = vcat([S_T_Δ ^ k for k in 0:p], [model.λ for _ in 1:m])
        y[i] = exp(-model.μ * model.T) * option(S_T)
    end
    return X, y, Nvec
end







radon_nikodym(λ₀, λ, N) = exp(λ₀ - λ) * (λ / λ₀) .^N

function sim_price_vibrato(model::MertonModel, option::OptionContract, λ₀::Real, nsim::Int, Ydist = LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2)))
    Random.seed!(42) # for reproducibility
    X,y,Nvec = Vibrato(model, option, λ₀, Ydist, nsim, 2, 1)
    bhat = X\y
    yfit = X * bhat
    return mean(yfit .* radon_nikodym.(λ₀ * (model.T - model.dt), model.λ * (model.T - model.dt), Nvec)), std(yfit .* radon_nikodym.(λ₀ * (model.T - model.dt), model.λ * (model.T - model.dt), Nvec)) / sqrt(nsim)
end

sim_price_vibrato(MertonModel(100.0, 1.0, 0.05, 0.2, 0.5, 1/252), CallOption(100.0), 0.5, 1_000_000)


# function simulated_price_and_sensitivity_vibrato(nsim::Int)
#     option = CallOption(100.0)
#     pr(λ) = sim_price_vibrato(MertonModel(100.0, 1.0, 0.05, 0.2, λ, 0.1), option, 0.5, nsim)
#     println("Simulated price and sensitivity for various λ values, using $nsim simulations:")
#     for λ in 0.0:0.1:1.0
#         m, se = pr(λ)
#         println("λ: $λ, Price: $(m) ($(se)), Sensitivity: $(ForwardDiff.derivative(x -> pr(x)[1], λ))")
#     end
# end

# simulated_price_and_sensitivity_vibrato(1_000_000)



