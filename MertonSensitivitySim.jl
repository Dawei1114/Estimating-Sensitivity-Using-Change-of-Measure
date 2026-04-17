using Distributions

function RadonNikodym(lambda0,lambda,N)
    return exp(lambda0 - lambda) * (lambda/lambda0)^N
end



