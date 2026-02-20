# Test0

## Model definition (mainly here to remind parameter names used in hetgpy)

For inputs $x\in\R^d$ output $f(x)\in \R$ and measurements $y\in\R$ model used has the parameters:
$$
\begin{aligned}
&\text{inverse lenghtscale: } &\theta &\in \R_+^d \\
&\text{nugget } &g &\in \R_+\\
&\text{covariance scale } &\nu &\in \R_+\\
&\text{trend } &b &\in \R\\
\end{aligned}
$$

i.e. with some (a lot) abuse of notation

$$
K_\theta(x, x') = e^{-||\theta(x-x')||^2} \\
f(x) \sim \mathcal{GP}(b, \nu K) \\
y | x \sim \mathcal{N}(f, \nu(K + g))
$$

>the function _kernel_ in gp.py directly implements $K$. I removed the option for Matérn kernels to simplify the code and debugging at this stage, but we should be able to plug and play any (differentiable) callable here.


## Mock problem definition

Define a relatively simple mock problem for testing.
$$
y=\cos(\pi x_1) + 0.1 \sin(3 \pi x_2) \\
\text{with: } x_1,x_2 \sim U(-1,1)
$$

This is a just a 2d function to make visualizations feasible.
There's a principal axis of low frequency, high amplitude changes and a secondary axis of higher frequency, lower amplitude change.

>open function_surface.html to see the function surface and the points used for train and testing of the models.


## Unregularized model and loss surface

The likelihood function we use to fit the model (gp.py -> likelihood) computes the likelihood of the parameters $\theta, g$ given the closed form optimum for $\nu, b$.

>open likelihood_contours.html for an interactive plot of the likelihood for different values of $\theta$ and $g$

Running ADMM with $\lambda=0$ is a good sanity check: 
the proximal operator for l1 becomes the identity, so the updated z always matches x, and therefore u=0.
So the optimization loop becomes a series of L-BFGS-B calls each solving the mle estimation with l2 penalty on deviations from the last iterate.

>open optimization_trajectoty.html to see the results of running admm on the unregularized problem

In the plots, I'm showing the admm loss heatmap for theta, given the nugett found after each step. 
The optimization gets trapped in a local minima.
I did not create this example on purpose to show this. Apparently this is very likely to happen. 
Before diagnosing the source of the problem, I was playing around with different variations of simple examples functions, and all ended up having the same issue of getting trapped in local minimas following hetgpy initialization.


# Test 1

## Testing L1 regularization

The non ideal initialization makes the algorithm fail in the unregularized problem. Adding regularization can only make this problem worse.
However it makes sense to sanity check that the algorithm is running correctly (even if it gets stuck in a local minima at times).
Indeed this is what happens. If the regularization is strong enough, the parameters are pushed to zero, but when the regularization is low and the optima would be a non-sparse solution, the algorithm gets stuck in a sparse local minima (the same of the non penalized model).

> you can see this effect in the _effect_of_regularization_warm_start=False.html_ file. NOTE: The optimization trajectories saved do not match this experiment, to see them just rerun test1.py with the desired flags. 

## Using warm starts
If we use warm start, the results are horrible. Doing this the naive way I was testing at the begginning (starting from low regularization and slowing increasing it) has the same exact issue of using the fixed initialization of hetgpy. It actually make the issue even more apparent, but at least it runs slightly faster.

> The results saved are not from this warm start setup. The results is very similar to having no warm star at all. To see this effect rerun _test1.py_ using flag _reverse_warm_start_order=False_.

Doing it in reverse (starting from high regularization and slowly decreasing it) completely breaks down. In this case, the algorithm gets stuck at zero and is never able to escape.
It is really hard to see what's happening unless we move to a log-scale for the plots of the optimization trajectories.
Basically what is happening is that in a large neighboorhood of $\theta=0$ the loss is effectively flat, so if the algorithm gets initialized to zero, it never escapes.
I did check the gradients directly, but did not also include that visualization since this part is already a bit messy. 
This effect gets more pronounced for large values of $g$.

> The saved files _optimization_trajectory_*.html_ show the results of this (use log scale to highlight the effect)

