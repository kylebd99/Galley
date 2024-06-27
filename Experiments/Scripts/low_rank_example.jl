using Finch
using Galley
using Galley: t_undef

function conjugate_gradient!(
    A::AbstractMatrix, b::AbstractVector, x::AbstractVector; tol=eps(eltype(b))
)
    # Initialize residual vector
    residual = b - A * x
    # Initialize search direction vector
    search_direction = residual
    # Compute initial squared residual norm
	norm(x) = sqrt(sum(x.^2))
    old_resid_norm = norm(residual)

    # Iterate until convergence
    while old_resid_norm > tol
        A_search_direction = A * search_direction
        step_size = old_resid_norm^2 / (search_direction' * A_search_direction)
        # Update solution
        @. x = x + step_size * search_direction
        # Update residual
        @. residual = residual - step_size * A_search_direction
        new_resid_norm = norm(residual)

        # Update search direction vector
        @. search_direction = residual +
            (new_resid_norm / old_resid_norm)^2 * search_direction
        # Update squared residual norm for next iteration
        old_resid_norm = new_resid_norm
    end
    return x
end

function run_exps()
    A = B * C
    # Initialize residual vector
    residual = b - A * x
    # Initialize search direction vector
    search_direction = residual
    # Compute initial squared residual norm
    norm(x) = sqrt(sum(x.^2))
    old_resid_norm = norm(residual)

    # Iterate until convergence
    while old_resid_norm > tol
        A_search_direction = A * search_direction
        step_size = old_resid_norm^2 / (search_direction' * A_search_direction)
        # Update solution
        @. x = x + step_size * search_direction
        # Update residual
        @. residual = residual - step_size * A_search_direction
        new_resid_norm = norm(residual)

        # Update search direction vector
        @. search_direction = residual +
            (new_resid_norm / old_resid_norm)^2 * search_direction
        # Update squared residual norm for next iteration
        old_resid_norm = new_resid_norm
    end
end

function galley_iter_step(B, C, X)
    A_g = Materialize(t_undef, t_undef, :i, :k, Σ(:j, MapJoin(*, B[:i,:j], C[:k,:j])))
    residual = Materialize(t_undef, :i, Σ(:j, MapJoin(*, A_g[:i,:j], X[:j])))
    old_resid_norm = Materialize(Σ(:i, MapJoin(*, residual[:i], residual[:i])))
    search_direction = residual
    A_search_direction = Materialize(t_undef, :i, Σ(:j, MapJoin(*, A_g[:i, :j], search_direction[:j])))
    step_size = Materialize(MapJoin(/, MapJoin(^, old_resid_norm[], 2),
                            Σ(:i, MapJoin(*, search_direction[:i], A_search_direction[:i]))))
    new_x = Materialize(t_undef, :i, MapJoin(+, X[:i], MapJoin(*, step_size[], search_direction[:i])))
    new_residual = Materialize(t_undef, :i, MapJoin(+, MapJoin(-, residual[:i]), MapJoin(*, step_size[], A_search_direction[:i])))
    new_residual_norm = Materialize(Σ(:i, MapJoin(*, Alias(:new_residual, :i), Alias(:new_residual, :i))))
    new_search_direction = Materialize(t_undef, :i, MapJoin(+, Alias(:new_residual, :i), MapJoin(*, MapJoin(^, MapJoin(/, Alias(:new_residual_norm), old_resid_norm[]), 2), search_direction[:i])))

    queries = [Query(:X, new_x),
                Query(:new_residual, new_residual),
                Query(:new_residual_norm, new_residual_norm),
                Query(:new_search_direction, new_search_direction)]
    println(queries)
    return galley(queries, verbose=4)
end

n,k,m = (1000, 10, 1000)
B = Materialize(t_undef, t_undef, :i, :j, Input(Tensor(Dense(Sparse(Element(0.0))), fsprand(Int, n, k, 100)), :i, :j))
C = Materialize(t_undef, t_undef, :i, :j, Input(Tensor(Dense(Sparse(Element(0.0))), fsprand(Int, m, k, 100)), :i, :j))
X = Materialize(t_undef, :i, Input(Tensor(Dense(Element(0.0)), rand(Int, m) .% 100), :i))
galley_iter_step(B, C, X)
