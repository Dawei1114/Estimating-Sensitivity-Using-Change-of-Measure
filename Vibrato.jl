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

function Vibrato(model::MertonModel, option::OptionContract, Ydist, nsims::Int, p::Int)
    Random.seed!(42) # for reproducibility
    Δ= model.dt
    θ = rand(nsims)
    X = Matrix{Float64}(undef,nsims, p + 2)
    y = Vector{Float64}(undef, nsims) # payoffs
    for i in eachindex(θ)
        stocks = MJD_FixedJumps(MertonModel(model.S0, model.T, model.μ, model.σ, θ[i], Δ), Ydist)
        S_T = exp(-model.μ * model.T) * option(stocks[end])
        S_T_Δ = stocks[end - 1]
        X[i,:] = vcat([S_T_Δ ^ k for k in 0:p], θ[i])
        y[i] = S_T
    end
    return X, y
end

Xtrain, ytrain = Vibrato(MertonModel(100,1,0.05,0.2,0.5,1/252), CallOption(100), LogNormal(-0.5 * log(1+0.1 ^ 2), log(1+0.1 ^ 2)), 1000, 2)

print("First 5 rows of the design matrix Xtrain:\n")
print("Cons | S_T_Δ | S_T_Δ^2 | θ\n")


# Display the first 5 rows of the design matrix Xtrain and the first 5 elements of the target vector ytrain
display(Xtrain[1:5, :])
print("First 5 elements of the target vector ytrain:\n")
print("E(f(S_T))\n")
display(ytrain[1:5])


bhat = Xtrain \ ytrain
e = ytrain - Xtrain * bhat
SE = sqrt.(diag(inv(Xtrain' * Xtrain) * (e'e/(length(ytrain) - length(bhat))))) # Standard errors of the coefficients
tstats = bhat ./ SE # t-statistics for the coefficients
print("Estimated coefficients (bhat):\n")
print("Cons | S_T_Δ | S_T_Δ^2 | θ\n")
for i in eachindex(bhat)
    println("bhat[$i]: $(bhat[i]), SE: $(SE[i]), tstat: $(tstats[i])")
end



# Estimated E(f(S_T)) = bhat[1] + bhat[2] * S_T_Δ + bhat[3] * S_T_Δ^2 + bhat[4] * θ


