import juliacall
import numpy as np

jl = juliacall.newmodule("PyDito")

jl.seval("using DifferentiableTrajectoryOptimization: DifferentiableTrajectoryOptimization as Dito")

cost = jl.seval(
"""
(xs, us, params) -> sum(sum((x - params).^2) + sum(u.^2) for (x, u) in zip(xs, us))
"""
)
dynamics = jl.seval(
"""
(x, u, t) -> x + u
"""
)

inequality_constraints = jl.seval(
"""
(xs, us, params) -> []
"""
)

state_dim = 2
control_dim = 2
parameter_dim = 2
horizon = 10

problem = jl.Dito.ParametricTrajectoryOptimizationProblem(
    cost,
    dynamics,
    inequality_constraints,
    state_dim,
    control_dim,
    parameter_dim,
    horizon
)

backend = jl.Dito.QPSolver()
optimizer = jl.Dito.Optimizer(problem, backend)

x0 = np.zeros(state_dim)
params = np.random.rand(parameter_dim)
solution = optimizer(x0, params)

print("states:")
solution.xs._jl_display()
print("controls:")
solution.us._jl_display()
