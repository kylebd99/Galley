# This function traverses the logical expression tree and converts it into a tensor expression
# tree for use in a tensor kernel. In doing so, it decides the boundaries of each kernel as
# well.
function _recursive_get_kernel_root(n, loop_order, input_counter)
    kernel_root = nothing
    input_dict = Dict()
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        def = deepcopy(get_def(n.stats))
        input_indices = relative_sort(collect(def.index_set), reverse(loop_order))
        stats = deepcopy(n.stats)
        def = get_def(stats)
        def.index_order = input_indices
        input_dict[next_tensor_id] = expr_to_kernel(n, input_indices)
        kernel_root = InputExpr(next_tensor_id, input_indices, [t_walk for _ in input_indices], stats)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_kernel_root, l_input_dict = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_kernel_root, r_input_dict = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        kernel_root = OperatorExpr(op, [l_kernel_root, r_kernel_root])
        input_dict = Dict(l_input_dict..., r_input_dict...)

    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_dict[next_tensor_id] = transpose_input(loop_order, n.args[2],  n.stats)
        stats = deepcopy(n.stats)
        def = get_def(stats)
        def.index_order = relative_sort(def.index_order, reverse(loop_order))
        kernel_root = InputExpr(next_tensor_id, def.index_order, [t_walk for _ in def.index_order], stats)
    end
    return kernel_root, input_dict
end

function _recursive_get_stats(n, input_counter)
    input_stats = Dict()
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_stats[next_tensor_id] = n.stats
    elseif n.head == MapJoin
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_input_stats = _recursive_get_stats(left_expr, input_counter)
        r_input_stats = _recursive_get_stats(right_expr, input_counter)
        input_stats = Dict(l_input_stats..., r_input_stats...)
    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_stats[next_tensor_id] = n.stats
    end
    return input_stats
end

# This function takes in a dict of (tensor_id => tensor_stats) and outputs a join order.
# Currently, it uses a simple prefix-join heuristic, but in the future it will be cost-based.
# TODO: Make a cost-based implementation of this function.
function get_join_loop_order_simple(input_stats)
    num_occurrences = counter(IndexExpr)
    for stats in values(input_stats)
        for v in get_index_set(stats)
            inc!(num_occurrences, v)
        end
    end
    vars_and_counts = sort([(num_occurrences[v], v) for v in keys(num_occurrences)], by=(x)->x[1], rev=true)
    vars = [x[2] for x in vars_and_counts]
    return vars
end


function get_join_loop_order(input_stats::Vector{TensorStats})
    all_vars = union([get_index_set(stat) for stat in input_stats]...)
    prefix_costs = Dict{Set{IndexExpr}, Float64}()
    optimal_prefix_orders = Dict{Set{IndexExpr}, Vector{IndexExpr}}()
    for var in all_vars
        v_set = Set([var])
        prefix_costs[v_set] = get_prefix_cost(v_set, input_stats, 0)
        optimal_prefix_orders[v_set] = [var]
    end

    for _ in 2:length(all_vars)
        new_prefix_costs = Dict{Set{IndexExpr}, Float64}()
        new_optimal_prefix_orders = Dict{Set{IndexExpr}, Vector{IndexExpr}}()
        for (prefix_set, min_cost) in prefix_costs
            # We only consider extensions that don't result in cross products
            potential_vars = Set{IndexExpr}()
            for stat in input_stats
                index_set = get_index_set(stat)
                if length(∩(index_set, prefix_set)) > 0
                    potential_vars = ∪(potential_vars, index_set)
                end
            end
            potential_vars = setdiff(potential_vars, prefix_set)
            for new_var in potential_vars
                new_prefix_set = union(prefix_set, [new_var])
                new_cost = get_prefix_cost(new_prefix_set, input_stats, min_cost)
                if !haskey(new_prefix_costs, new_prefix_set) || new_cost < new_prefix_costs[new_prefix_set]
                    new_prefix_costs[new_prefix_set] = new_cost
                    new_optimal_prefix_orders[new_prefix_set] = [optimal_prefix_orders[prefix_set]..., new_var]
                end
            end
        end
        prefix_costs = new_prefix_costs
        optimal_prefix_orders = new_optimal_prefix_orders
    end
    return optimal_prefix_orders[all_vars]
end

# This function takes in an input and replaces it with an input expression which matches the
# loop order of the kernel. This is essentially the same as building an index on the fly
# when needed.
function transpose_input(loop_order, input, stats)
    input_index_set = get_index_set(stats)
    transposed_index_order = reverse([x for x in loop_order if x in input_index_set])
    @assert !isnothing(get_index_order(stats))
    is_sorted = is_sorted_wrt_index_order(get_index_order(stats), transposed_index_order)
    if !is_sorted
        if input isa TensorKernel
            input.output_indices = transposed_index_order
        else
            @assert input isa Fiber
            input_indices = get_index_order(stats)
            expr = InputExpr("t_1", input_indices, [t_walk for _ in input_indices], stats)
            input_dict = Dict()
            input_dict["t_1"] = input
            expr = ReorderExpr(transposed_index_order, expr)
            output_formats = [t_hash for _ in 1:length(transposed_index_order)]
            output_dims = [get_dim_size(stats, idx) for idx in transposed_index_order]
            input = TensorKernel(expr,
                                    input_dict,
                                    transposed_index_order,
                                    output_formats,
                                    output_dims,
                                    get_default_value(stats),
                                    reverse(input_indices))
        end
    end
    return input
end



function select_output_format(output_stats::TensorStats,
                                loop_order::Vector{IndexExpr},
                                output_indices::Vector{IndexExpr}
                                )
    approx_sparsity = estimate_nnz(output_stats) / get_dim_space_size(get_def(output_stats), get_index_set(output_stats))
    if approx_sparsity > .5
        return [t_dense for _ in output_indices]
    end

    if is_prefix(output_indices, loop_order)
        return [t_sparse_list for _ in output_indices]
    end

    return [t_hash for _ in output_indices]
end


# This function takes in a logical plan and outputs a tree of tensor kernels.
# These kernels fully define the program that Finch will compile:
#   - Loop Order
#   - Output Format
#   - Access Protocols
# Currently, we decide the loop order entirely based on the structural properties of the
# query. Later on, we will use statistics and cost-based optimization for these decisions.
# Note, that we set the output order of the child to be the loop order of the parent.
# TODO: In the future, we would like to actually execute the kernels here and decide whether
# to re-optimize at this point based on the materialized children (e.g. their nnz count).
function expr_to_kernel(n::LogicalPlanNode, output_order::Vector{IndexExpr}; verbose = 0)
    kernel_root = nothing
    if n.head == Aggregate
        op = n.args[1]
        reduce_indices = n.args[2]
        sub_expr = n.args[3]
        input_stats = _recursive_get_stats(sub_expr, [1])
        input_stats_vec = Vector{TensorStats}(collect(values(input_stats)))
        loop_order = get_join_loop_order(input_stats_vec)
        output_indices = relative_sort(collect(get_index_set(n.stats)), output_order)
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        kernel_root = AggregateExpr(op, reduce_indices, body_kernel)
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        kernel = TensorKernel(kernel_root,
                                input_dict,
                                output_indices,
                                output_formats,
                                output_dims,
                                get_default_value(n.stats),
                                loop_order)
        validate_kernel(kernel)
        return kernel

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        input_counter = [1]
        l_input_stats = _recursive_get_stats(left_expr, input_counter)
        r_input_stats = _recursive_get_stats(right_expr, input_counter)
        input_stats = Dict(l_input_stats..., r_input_stats...)
        input_stats_vec = Vector{TensorStats}(collect(values(input_stats)))
        loop_order = get_join_loop_order(input_stats_vec)
        output_indices = relative_sort(collect(get_index_set(n.stats)), output_order)
        l_body_kernel, l_input_dict = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_body_kernel, r_input_dict = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        kernel_root = OperatorExpr(op, [l_body_kernel, r_body_kernel])
        input_dict = Dict(l_input_dict..., r_input_dict...)
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        kernel = TensorKernel(kernel_root,
                                input_dict,
                                output_indices,
                                output_formats,
                                output_dims,
                                get_default_value(n.stats),
                                loop_order)
        validate_kernel(kernel)
        return kernel

    elseif n.head == Reorder
        sub_expr = n.args[1]
        output_indices = n.args[2]
        loop_order = reverse(output_indices)
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        kernel_root = ReorderExpr(output_indices, body_kernel)
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        kernel = TensorKernel(kernel_root,
                                input_dict,
                                output_indices,
                                output_formats,
                                output_dims,
                                get_default_value(n.stats),
                                loop_order)
        validate_kernel(kernel)
        return kernel

    elseif n.head == InputTensor
        return n.args[2]

    elseif n.head == Scalar
        return n.args[1]
    end
end



# This function does a variety of sanity checks on the kernel before we attempt to execute it.
# Such as:
#  1. Check that the loop order is a permutation of the input indices
#  2. Check that the output indices are the inputs minus any that are aggregate_indices
#  3. Check that the inputs are all sorted w.r.t. the loop order
function validate_kernel(kernel::TensorKernel)
    println(kernel.output_formats)
    function get_input_indices(n::TensorExpression)
        return if n isa InputExpr
            Set(n.input_indices)
        elseif n isa AggregateExpr
            get_input_indices(n.input)
        elseif n isa OperatorExpr
            union([get_input_indices(input) for input in n.inputs]...)
        elseif n isa ReorderExpr
            get_input_indices(n.input)
        end
    end
    input_indices = get_input_indices(kernel.kernel_root)
    @assert  input_indices == Set(kernel.loop_order)

    function get_output_indices(n::TensorExpression)
        return if n isa InputExpr
            Set(n.input_indices)
        elseif n isa AggregateExpr
            setdiff(get_input_indices(n.input), n.aggregate_indices)
        elseif n isa OperatorExpr
            union([get_input_indices(input) for input in n.inputs]...)
        elseif n isa ReorderExpr
            get_input_indices(n.input)
        end
    end
    output_indices = get_output_indices(kernel.kernel_root)
    @assert output_indices ⊆ input_indices
    @assert length(output_indices) == length(kernel.output_formats)

    function check_sorted_inputs(n::TensorExpression)
        if n isa InputExpr
            @assert is_sorted_wrt_index_order(n.input_indices, kernel.loop_order; loop_order=true)
        elseif n isa AggregateExpr
            check_sorted_inputs(n.input)
        elseif n isa OperatorExpr
            for input in n.inputs
                check_sorted_inputs(input)
            end
        elseif n isa ReorderExpr
            check_sorted_inputs(n.input)
        end
    end
    check_sorted_inputs(kernel.kernel_root)
end
