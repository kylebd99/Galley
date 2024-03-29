MAX_KERNEL_SIZE = 6

function get_tensor_id(input_counter)
    id = "t_$(input_counter[1])"
    input_counter[1] += 1
    return id
end

function _recursive_get_stats(n)
    input_stats = []
    if n.head == Aggregate || n.head == Reorder
        push!(input_stats, n.stats)
    elseif n.head == MapJoin
        children = n.args[2:end]
        for child in children
            append!(input_stats, _recursive_get_stats(child))
        end
    elseif n.head == InputTensor || n.head == Scalar
        push!(input_stats, n.stats)
    end
    return input_stats
end

KI = NamedTuple{(:node, :stats, :root, :input_dict, :input_exprs), Tuple{LogicalPlanNode, TensorStats, TensorExpression, Dict, Vector{InputExpr}}}
KI_V = NamedTuple{(:children, :reduce_vars), Tuple{Vector, Set{IndexExpr}}}

# Takes in a list of statistics objects and produces a tree of lists which represent
# a good way to break down the mapjoin.
function make_input_tree(sum_op, mult_op, child_kernel_info, overall_reduce_vars)
    input_tree = KI_V[(children=KI[ kernel_info ], reduce_vars=Set{IndexExpr}()) for kernel_info in child_kernel_info]
    cur_stats = [kernel_info.stats for kernel_info in child_kernel_info]
    occurences = [0 for stats in cur_stats]
    outermost_occurences = sum([length(get_index_set(stat)) for stat in cur_stats]) - length(union([get_index_set(stat) for stat in cur_stats]...))
    while outermost_occurences > MAX_KERNEL_SIZE && length(input_tree) > 2
        min_cost = Inf
        best_pair = (-1, -1)
        for (i, i_stats) in enumerate(cur_stats)
            for (j, j_stats) in enumerate(cur_stats)
                i >= j && continue # Only need to compare each pair once
                overlapping_vars = length(union(length(get_index_set(i_stats)), length(get_index_set(j_stats))))
                min_next_occurrences = min(occurences[i] + occurences[j] + overlapping_vars,
                                            occurences[i] + overlapping_vars,
                                            occurences[j] + overlapping_vars)
                min_next_occurrences > MAX_KERNEL_SIZE && continue
                new_reduce_vars = setdiff(overall_reduce_vars, input_tree[i][2], input_tree[j][2])
                new_reduce_vars = Set{IndexExpr}(setdiff(new_reduce_vars, [get_index_set(stat) for (k, stat) in enumerate(cur_stats) if k!=i && k!=j]...))
                new_stats = reduce_tensor_stats(sum_op, new_reduce_vars, merge_tensor_stats(mult_op, i_stats, j_stats))
                condense_stats!(new_stats)
                cost = estimate_nnz(new_stats)
                if min_cost > cost
                    min_cost = cost
                    best_pair = (i,j)
                end
            end
        end
        min_cost == Inf && (println("MIN COST INF"); break)
        i = best_pair[1]
        j = best_pair[2]
        new_expr_group = []
        new_reduce_vars = Set{IndexExpr}(setdiff(overall_reduce_vars, input_tree[i][2], input_tree[j][2]))
        new_reduce_vars = setdiff(new_reduce_vars, [get_index_set(stat) for (k, stat) in enumerate(cur_stats) if k!=i && k!=j]...)
        overlapping_vars = length(union(length(get_index_set(cur_stats[i])), length(get_index_set(cur_stats[j]))))
        new_occurence = 0
        if occurences[i] + occurences[j] + overlapping_vars <= MAX_KERNEL_SIZE
            new_expr_group = [input_tree[i][1]..., input_tree[j][1]...]
            new_reduce_vars = ∪(new_reduce_vars, input_tree[i][2], input_tree[j][2])
            new_occurence = occurences[i] + occurences[j] + overlapping_vars
        elseif occurences[i] < occurences[j]
            new_expr_group = [input_tree[i][1]..., input_tree[j]]
            new_reduce_vars = ∪(new_reduce_vars, input_tree[i][2])
            new_occurence = occurences[i] + overlapping_vars
        else
            new_expr_group = [input_tree[i], input_tree[j][1]...]
            new_reduce_vars = ∪(new_reduce_vars, input_tree[j][2])
            new_occurence = overlapping_vars + occurences[j]
        end
        input_tree = KI_V[grp for (k, grp) in enumerate(input_tree) if k!=i && k!=j]
        new_reduce_vars = Set{IndexExpr}(new_reduce_vars)
        new_expr_group = (children=new_expr_group, reduce_vars=new_reduce_vars)
        push!(input_tree, new_expr_group)
        new_stats = merge_tensor_stats(mult_op, cur_stats[i], cur_stats[j])
        if length(new_reduce_vars) > 0
            new_stats = reduce_tensor_stats(sum_op, new_reduce_vars, new_stats)
        end
        cur_stats = [stat for (k, stat) in enumerate(cur_stats) if k!=i && k!=j]
        push!(cur_stats, new_stats)
        occurences = [count for (k, count) in enumerate(occurences) if k!=i && k!=j]
        push!(occurences, new_occurence)
        outermost_occurences = sum([length(get_index_set(stat)) for stat in cur_stats]) - length(union([get_index_set(stat) for stat in cur_stats]...))
    end
    return input_tree
end

function _process_input_tree(sum_op, mult_op, input_tree, overall_output_order, input_counter, reduce_vars, verbose)
    child_nodes = []
    child_stats = []
    overall_removed_vars = Set{IndexExpr}(reduce_vars)
    for child in input_tree
        if child isa KI
            push!(child_nodes, child.node)
            push!(child_stats, child.stats)
        else
            child_inputs = child[1]
            child_reduce_vars = child[2]
            child_node, child_removed_vars = _process_input_tree(sum_op, mult_op, child_inputs, overall_output_order, input_counter, child_reduce_vars, verbose)
            push!(child_nodes, child_node)
            push!(child_stats, child_node.stats)
            union!(overall_removed_vars, child_removed_vars)
        end
    end
    expr = MapJoin(mult_op, child_nodes...)
    expr.stats = merge_tensor_stats(mult_op, child_stats...)
    kernel = nothing
    if length(reduce_vars) > 0
        reduce_vars = Set{IndexExpr}(reduce_vars)
        agg_stats = reduce_tensor_stats(sum_op, reduce_vars, expr.stats)
        expr = Aggregate(sum_op, reduce_vars, expr)
        expr.stats = agg_stats
        expr_idxs = get_index_set(expr.stats)
        kernel = aggregate_to_kernel(expr, [idx for idx in overall_output_order if idx in expr_idxs], verbose)
    else
        expr_idxs = get_index_set(expr.stats)
        kernel = mapjoin_to_kernel(expr, [idx for idx in overall_output_order if idx in expr_idxs], verbose)
    end
    def = get_def(expr.stats)
    def.index_order = kernel.output_indices
    def.level_formats = kernel.output_formats
    node = LogicalPlanNode(InputTensor, [kernel.output_indices, kernel], expr.stats)
    return node, overall_removed_vars
end

# This function traverses the logical expression tree and converts it into a tensor expression
# tree for use in a tensor kernel. In doing so, it decides the boundaries of each kernel as
# well.
function _recursive_get_kernel_root(n, loop_order, input_counter, parent, verbose)
    kernel_root = nothing
    input_dict = Dict()
    input_exprs = []
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = get_tensor_id(input_counter)
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
        args = n.args[2:end]
        child_kernel_info = KI[]
        for arg in args
            root, dict, exprs = _recursive_get_kernel_root(arg, loop_order, input_counter, n, verbose)
            push!(child_kernel_info, (node=arg, stats=arg.stats, root=root, input_dict=dict, input_exprs=exprs))
        end
        idx_occurences = sum([length(get_index_set(ki.stats)) for ki in child_kernel_info])
        if idx_occurences > MAX_KERNEL_SIZE
            reduced_vars = IndexExpr[]
            sum_op = +
            if parent.head == Aggregate
                sum_op = parent.args[1]
                reduced_vars = parent.args[2]
            end
            input_tree = make_input_tree(sum_op, op, child_kernel_info, reduced_vars)
            child_roots = []
            input_dict = Dict()
            input_exprs = []
            output_order = relative_sort(collect(get_index_set(n.stats)), reverse(loop_order))
            removed_vars = Set{IndexExpr}()
            for (inputs, input_reduce_vars) in input_tree
                if length(inputs) > 1
                    child_node, child_removed_vars = _process_input_tree(sum_op, op, inputs, output_order, input_counter, input_reduce_vars, verbose)
                    child_kernel = child_node.args[2]
                    child_expr = InputExpr(get_tensor_id(input_counter),
                                            child_kernel.output_indices,
                                            [t_default for _ in child_kernel.output_indices],
                                            child_node.stats)
                    input_dict[child_expr.tensor_id] = child_kernel
                    push!(child_roots, child_expr)
                    push!(input_exprs, child_expr)
                    union!(removed_vars, child_removed_vars)
                else
                    ki = inputs[1]
                    merge!(input_dict, ki.input_dict)
                    append!(input_exprs, ki.input_exprs)
                    push!(child_roots, ki.root)
                end
            end
            if parent.head == Aggregate
                setdiff!(parent.args[2], removed_vars)
            end
            kernel_root = OperatorExpr(op, [child_roots...])
        else
            kernel_root = OperatorExpr(op, [[ki.root for ki in child_kernel_info]...])
            input_dict = merge([ki.input_dict for ki in child_kernel_info]...)
            input_exprs = union([ki.input_exprs for ki in child_kernel_info]...)
        end
    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = get_tensor_id(input_counter)
        input_counter[1] += 1
        input_dict[next_tensor_id], tp_stats = transpose(reverse(loop_order), n.args[2],  n.stats)
        kernel_root = InputExpr(next_tensor_id, get_index_order(tp_stats), [t_default for idx in get_index_order(tp_stats)], tp_stats)
        push!(input_exprs, kernel_root)
    end
    return kernel_root, input_dict, input_exprs
end

function aggregate_to_kernel(n::LogicalPlanNode, output_order::Vector{IndexExpr}, verbose)
    op = n.args[1]
    reduce_indices = n.args[2]
    starting_reduce_indices = copy(reduce_indices)
    sub_expr = n.args[3]
    input_stats_vec = Vector{TensorStats}(_recursive_get_stats(sub_expr))
    loop_order = get_join_loop_order(input_stats_vec, n.stats, output_order)
    body_kernel, input_dict, input_exprs = _recursive_get_kernel_root(sub_expr, loop_order, [1], n, verbose)
    already_reduced_indices = setdiff(starting_reduce_indices, reduce_indices)
    setdiff!(loop_order, already_reduced_indices)
    modify_protocols!(input_exprs)
    kernel_root = AggregateExpr(op, reduce_indices, body_kernel)
    # Based on the loop order, we attempt to avoid random writes into the output.
    output_indices = if set_compat_with_loop_prefix(get_index_set(n.stats), loop_order)
        relative_sort(collect(get_index_set(n.stats)), reverse(loop_order))
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
    kernel, tp_stats = transpose(output_order, kernel, n.stats)
    verbose >= 2 && validate_kernel(kernel)
    return kernel
end

function mapjoin_to_kernel(n::LogicalPlanNode, output_order::Vector{IndexExpr}, verbose)
    op = n.args[1]
    children = n.args[2:end]
    input_stats_vec = TensorStats[]
    input_counter = [1]
    for child in children
        append!(input_stats_vec, _recursive_get_stats(child))
    end
    input_roots = TensorExpression[]
    input_exprs = []
    input_counter = [1]
    input_dict = Dict()
    loop_order = get_join_loop_order(input_stats_vec, n.stats, output_order)
    for child in children
        child_root, child_input_dict, child_exprs = _recursive_get_kernel_root(child, loop_order, input_counter, n, verbose)
        input_dict = merge(input_dict, child_input_dict)
        input_exprs = [input_exprs..., child_exprs...]
        push!(input_roots, child_root)
    end
    modify_protocols!(input_exprs)
    kernel_root = OperatorExpr(op, input_roots)
    # Based on the loop order, we attempt to avoid random writes into the output.
    output_indices = if set_compat_with_loop_prefix(get_index_set(n.stats), loop_order)
        relative_sort(collect(get_index_set(n.stats)), reverse(loop_order))
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
    kernel, tp_stats = transpose(output_order, kernel, n.stats)
    verbose >= 2 && validate_kernel(kernel)
    return kernel
end

function reorder_to_kernel(n::LogicalPlanNode, output_order::Vector{IndexExpr}, verbose)
    sub_expr = n.args[1]
    output_indices = n.args[2]
    loop_order = reverse(output_indices)
    body_kernel, input_dict, input_exprs = _recursive_get_kernel_root(sub_expr, loop_order, [1], n, verbose)
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
    verbose >= 2 && validate_kernel(kernel)
    return kernel
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
    if n.head == Aggregate
        return aggregate_to_kernel(n, output_order, verbose)
    elseif n.head == MapJoin
        return mapjoin_to_kernel(n, output_order, verbose)
    elseif n.head == Reorder
        return reorder_to_kernel(n, output_order, verbose)
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
