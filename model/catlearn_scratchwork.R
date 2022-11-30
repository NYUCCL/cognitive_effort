library(catlearn)

data(nosof94) # load data
nosof94plot(nosof94,title = "nosofsky et al. 94")

nosof94sustain(params = c(9.01245, 1.252233, 16.924073, 0.092327))

nosof94plot(.Last.value, title = "sustain")
