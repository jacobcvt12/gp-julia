using Distributions, Distances

# squared exponential covariance function
function K(X_p, X_q)
    return 1
end

# set seed
srand(1)

x_star = -5:0.1:5
K_psd = exp(-0.5 * pairwise(SqEuclidean(), x_star'))
K_pd = copy(K);
for i in 1:length(x_star)
    K_pd[i, i] = K_pd[i, i] + 0.001
end
det(K_pd)
det(K_psd)
f_star = rand(MvNormal(K_pd), 1)

using Gadfly

plot(x=x_star, y=f_star[:, ], Geom.point, Geom.line)
