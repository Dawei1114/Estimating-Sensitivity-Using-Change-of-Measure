using Distributions, Statistics

var = 0.1
mu = -var/2
Y_test = rand(LogNormal(mu, var), 1_000_000)
mean(Y_test)
