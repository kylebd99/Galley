# This function traverses the logical expression tree and converts it into a tensor expression
# tree for use in a tensor kernel. In doing so, it decides the boundaries of each kernel as
# well.
function _recursive_get_kernel_root(n, loop_order, input_counter)
    kernel_root = nothing
    input_dict = Dict()
    input_exprs = InputExpr[]
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        def = get_def(n.stats)
        input_indices = relative_sort(collect(def.index_set), reverse(loop_order))
        input_dict[next_tensor_id] = expr_to_kernel(n, input_indices)
        agg_stats = deepcopy(n.stats)
        agg_def = get_def(agg_stats)
        agg_def.index_order = input_indices
        agg_def.level_formats = input_dict[next_tensor_id].output_formats
        kernel_root = InputExpr(next_tensor_id, input_indices, [t_default for idx in input_indices], agg_stats)
        push!(input_exprs, kernel_root)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_kernel_root, l_input_dict, l_input_exprs = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_kernel_root, r_input_dict, r_input_exprs = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        kernel_root = OperatorExpr(op, [l_kernel_root, r_kernel_root])
        input_dict = Dict(l_input_dict..., r_input_dict...)
        input_exprs = [l_input_exprs..., r_input_exprs...]

    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_dict[next_tensor_id], tp_stats = transpose_input(loop_order, n.args[2],  n.stats)
        kernel_root = InputExpr(next_tensor_id, get_index_order(tp_stats), [t_default for idx in get_index_order(tp_stats)], tp_stats)
        push!(input_exprs, kernel_root)
    end
    return kernel_root, input_dict, input_exprs
end

function select_leader_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_walk
    elseif format == t_dense
        return t_default
    elseif format == t_hash
        return t_default
    end
end

function select_follower_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_default
    elseif format == t_dense
        return t_follow
    elseif format == t_hash
        return t_default
    end
end

function modify_protocols!(input_exprs::Vector{InputExpr})
    vars = union([i.input_indices for i in input_exprs]...)
    for var in vars
        relevant_inputs = [i for i in input_exprs if var ∈ i.input_indices]
        costs = []
        for input in relevant_inputs
            if get_index_format(input.stats, var) == t_dense
                push!(costs, get_dim_size(input.stats, var))
                continue
            end
            size_before_var = 1
            indices_before_var = []
            for index in input.input_indices
                index == var && break
                push!(indices_before_var, index)
            end
            if length(indices_before_var) > 0
                size_before_var = estimate_nnz(reduce_tensor_stats(+, setdiff(Set(input.input_indices), indices_before_var),  input.stats))
            end
            size_after_var = estimate_nnz(reduce_tensor_stats(+, setdiff(Set(input.input_indices), [indices_before_var..., var]),  input.stats))
            push!(costs, size_after_var/size_before_var)
        end
        min_cost = minimum(costs)
        needs_leader = length(relevant_inputs) > 1
        formats = [get_index_format(input.stats, var) for input in relevant_inputs]
        num_sparse_lists = sum([f == t_sparse_list for f in formats])
        use_gallop = false
        if num_sparse_lists > 1
            gallop_cost = minimum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * RandomReadCost
            walk_cost = maximum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * SeqReadCost
            use_gallop = gallop_cost < walk_cost
        end
        for i in eachindex(relevant_inputs)
            input = relevant_inputs[i]
            var_index = findfirst(x->x==var, input.input_indices)
            is_leader = costs[i] == min_cost

            if formats[i] == t_sparse_list
                if use_gallop
                    input.input_protocols[var_index] = t_gallop
                else
                    input.input_protocols[var_index] = t_walk
                end
                needs_leader = false
                continue
            end

            if is_leader && needs_leader
                input.input_protocols[var_index] = select_leader_protocol(formats[i])
                needs_leader = false
            else
                input.input_protocols[var_index] = select_follower_protocol(formats[i])
            end
        end
    end
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


# This function takes in an input and replaces it with an input expression which matches the
# loop order of the kernel. This is essentially the same as building an index on the fly
# when needed.
function transpose_input(loop_order, input, stats)
    tp_stats = deepcopy(stats)
    input_index_set = get_index_set(tp_stats)
    transposed_index_order = reverse([x for x in loop_order if x in input_index_set])
    @assert !isnothing(get_index_order(tp_stats))
    is_sorted = is_sorted_wrt_index_order(get_index_order(tp_stats), transposed_index_order)
    if !is_sorted
        @assert input isa Tensor
        input_indices = get_index_order(tp_stats)
        expr = InputExpr("t_1", input_indices, [t_default for _ in input_indices], tp_stats)
        input_dict = Dict()
        input_dict["t_1"] = input
        expr = ReorderExpr(transposed_index_order, expr)
        output_formats = [t_hash for _ in 1:length(transposed_index_order)]
        output_dims = [get_dim_size(tp_stats, idx) for idx in transposed_index_order]
        input = TensorKernel(expr,
                                input_dict,
                                transposed_index_order,
                                output_formats,
                                output_dims,
                                get_default_value(tp_stats),
                                reverse(input_indices))

        expr = InputExpr("t_1", transposed_index_order, [t_default for _ in transposed_index_order], tp_stats)
        input_dict = Dict()
        input_dict["t_1"] = input
        expr = ReorderExpr(transposed_index_order, expr)
        output_formats = [t_sparse_list for _ in 1:length(transposed_index_order)]
        output_formats[length(output_formats)] = t_dense
        output_dims = [get_dim_size(tp_stats, idx) for idx in transposed_index_order]
        input = TensorKernel(expr,
                                input_dict,
                                transposed_index_order,
                                output_formats,
                                output_dims,
                                get_default_value(tp_stats),
                                reverse(transposed_index_order))
        def = get_def(tp_stats)
        def.level_formats = output_formats
        def.index_order = transposed_index_order
    end
    return input, tp_stats
end

function transpose_kernel(output_order::Vector{IndexExpr}, kernel::TensorKernel, stats::TensorStats)
    transposed_index_order =  [x for x in output_order]
    is_sorted = is_sorted_wrt_index_order(kernel.output_indices, transposed_index_order)
    if is_sorted
        return kernel
    end
    input_indices = kernel.output_indices
    input_dict = Dict()
    input_dict["t_1"] = kernel
    new_stats = deepcopy(stats)
    def = get_def(new_stats)
    def.index_order = kernel.output_indices
    def.level_formats = kernel.output_formats
    expr = InputExpr("t_1", input_indices, [t_default for _ in input_indices], new_stats)
    expr = ReorderExpr(transposed_index_order, expr)
    output_formats = [t_hash for _ in 1:length(transposed_index_order)]
    output_dims = [get_dim_size(new_stats, idx) for idx in transposed_index_order]
    input = TensorKernel(expr,
                            input_dict,
                            transposed_index_order,
                            output_formats,
                            output_dims,
                            kernel.output_default,
                            reverse(input_indices))

    tp_stats = deepcopy(new_stats)
    expr = InputExpr("t_1", transposed_index_order, [t_default for _ in transposed_index_order], tp_stats)
    input_dict = Dict()
    input_dict["t_1"] = input
    expr = ReorderExpr(transposed_index_order, expr)
    output_formats = [t_sparse_list for _ in 1:length(transposed_index_order)]
    output_formats[length(output_formats)] = t_dense
    output_dims = [get_dim_size(tp_stats, idx) for idx in transposed_index_order]
    input = TensorKernel(expr,
                            input_dict,
                            transposed_index_order,
                            output_formats,
                            output_dims,
                            get_default_value(tp_stats),
                            reverse(transposed_index_order))
    def = get_def(tp_stats)
    def.level_formats = output_formats
    def.index_order = transposed_index_order
    return input
end

function select_output_format(output_stats::TensorStats,
                                loop_order::Vector{IndexExpr},
                                output_indices::Vector{IndexExpr}
                                )
    if length(output_indices) == 0
        return LevelFormat[]
    end

    approx_sparsity = estimate_nnz(output_stats) / get_dim_space_size(get_def(output_stats), get_index_set(output_stats))
    if approx_sparsity > .1
        return [t_dense for _ in output_indices]
    end

    formats = if is_prefix(output_indices, loop_order)
        [t_sparse_list for _ in output_indices]
    else
        [t_hash for _ in output_indices]
    end

    if length(formats) > 1
        formats[length(formats)] = t_dense
    end
    return formats
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
        loop_order = get_join_loop_order(input_stats_vec, n.stats, output_order)
        body_kernel, input_dict, input_exprs = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        modify_protocols!(input_exprs)
        kernel_root = AggregateExpr(op, reduce_indices, body_kernel)
        # Based on the loop order, we attempt to avoid random writes into the output.
        output_indices = if is_set_prefix(get_index_set(n.stats), loop_order)
            relative_sort(collect(get_index_set(n.stats)), loop_order)
        else
            relative_sort(collect(get_index_set(n.stats)), output_order)
        end
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        kernel = TensorKernel(kernel_root,
                                input_dict,
                                output_indices,
                                output_formats,
                                output_dims,
                                get_default_value(n.stats),
                                loop_order)
        kernel = transpose_kernel(output_order, kernel, n.stats)
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
        loop_order = get_join_loop_order(input_stats_vec, n.stats, output_order)
        l_body_kernel, l_input_dict, l_input_exprs = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_body_kernel, r_input_dict, r_input_exprs = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        modify_protocols!([l_input_exprs..., r_input_exprs...])
        kernel_root = OperatorExpr(op, [l_body_kernel, r_body_kernel])
        input_dict = Dict(l_input_dict..., r_input_dict...)
        # Based on the loop order, we attempt to avoid random writes into the output.
        output_indices = if is_set_prefix(get_index_set(n.stats), loop_order)
            relative_sort(collect(get_index_set(n.stats)), loop_order)
        else
            relative_sort(collect(get_index_set(n.stats)), output_order)
        end
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        kernel = TensorKernel(kernel_root,
                                input_dict,
                                output_indices,
                                output_formats,
                                output_dims,
                                get_default_value(n.stats),
                                loop_order)
        kernel = transpose_kernel(output_order, kernel, n.stats)
        validate_kernel(kernel)
        return kernel

    elseif n.head == Reorder
        sub_expr = n.args[1]
        output_indices = n.args[2]
        loop_order = reverse(output_indices)
        body_kernel, input_dict, input_exprs = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        kernel_root = ReorderExpr(output_indices, body_kernel)
        output_formats = select_output_format(n.stats, loop_order, output_indices)
        output_dims = [get_dim_size(n.stats, idx) for idx in output_indices]
        modify_protocols!([input_exprs...])
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
