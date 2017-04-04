using Distributions, Distances

# set seed
srand(1)

# sample from gaussian process

# covariate
x_star = -5:0.1:5

# squared exponential covariance function
K_psd = exp(-0.5 * pairwise(SqEuclidean(), x_star'))

# correct issue with K being singular
K_pd = copy(K_psd)

for i in 1:length(x_star)
    K_pd[i, i] = K_pd[i, i] + 0.001
end

f_star = rand(MvNormal(K_pd), 3)

# now plot draws from GP prior
using Gadfly, DataFrames
gp_draws = DataFrame(f_star)
gp_draws[:x_star] = x_star
gp_draws = stack(gp_draws, [:x1, :x2, :x3], [:x_star])

plot(gp_draws, x="x_star", y="value", Geom.point)
