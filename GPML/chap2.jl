using Distributions, Distances

# set seed
srand(1)

# sample from gaussian process

# covariate
x_star = -5:0.01:5

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

plot(gp_draws, x="x_star", y="value", color="variable", Geom.line)

# now draw from posterior gp given training points in 2.2
X = [-4, -3, -1, 0, 2]
y = [-2, 0, 1, 2, -1]
sigma_2_n = 0.1

function gp_posterior(X, y, x_star, sigma_2_n)
    K = exp(-0.5 * pairwise(SqEuclidean(), X'))
    L = chol(K + sigma_2_n * I)
    α = L' \ (L \ y)

    f_star = Array{Float64}(length(x_star))
    V_f_star = Array{Float64}(length(x_star))

    for i in 1:length(x_star)
        k_star = exp(-0.5 * pairwise(SqEuclidean(), X', [x_star[i]]'))
        f_star[i] = dot(k_star, α)
        v = L \ k_star
        V_f_star[i] = exp(-0.5 * pairwise(SqEuclidean(), [x_star[i]]')[1]) -
                      dot(v, v)
    end

    return f_star, V_f_star
end

f, v = gp_posterior(X, y, x_star, sigma_2_n)

gp_draw = [rand(Normal(fi[1], sqrt(fi[2])), 1)[1] for fi in zip(f, v)]
