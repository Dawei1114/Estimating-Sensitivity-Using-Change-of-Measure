using Distributions

mean = 1
var = 0.1
Y_test = rand(LogNormal(0, var), 1_000_000)
emp_mean = mean(Y_test)
emp_std_error = std(Y_test)/sqrt(1_000_000)
emp_var = var(Y_test)
prinln("Hello World!")