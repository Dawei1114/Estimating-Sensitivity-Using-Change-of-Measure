T = ((1,2,3),(4,5,6))

a1, a2 = map(x -> x .^ 2, T) # a1 and a2 will both be tuples of arrays, where each array is the square of the corresponding array in T

a1,a2