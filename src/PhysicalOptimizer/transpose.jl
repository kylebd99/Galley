
# This function takes in an input and replaces it with an input expression which matches the
# loop order of the kernel. This is essentially the same as building an index on the fly
# when needed.
function transpose(loop_order, input, stats)
    if input isa Tensor
        return transpose_input(loop_order, input, stats)
    elseif input isa TensorKernel
        return transpose_kernel(loop_order, input, stats)
    else
        throw(ArgumentError("Can only transpose kernels and tensors"))
    end
end

function transpose_input(loop_order, input, stats)
    initial_stats = deepcopy(stats)
    input_index_set = get_index_set(initial_stats)
    transposed_index_order = reverse([x for x in loop_order if x in input_index_set])
    @assert !isnothing(get_index_order(initial_stats))
    is_sorted = is_sorted_wrt_index_order(get_index_order(initial_stats), transposed_index_order)
    if !is_sorted
        @assert input isa Tensor
        input_indices = get_index_order(initial_stats)
        expr = InputExpr("t_1", input_indices, [t_default for _ in input_indices], initial_stats)
        input_dict = Dict()
        input_dict["t_1"] = input
        expr = ReorderExpr(transposed_index_order, expr)
        output_formats = [t_hash for _ in 1:length(transposed_index_order)]
        output_dims = [get_dim_size(initial_stats, idx) for idx in transposed_index_order]
        input = TensorKernel(expr,
                                input_dict,
                                transposed_index_order,
                                output_formats,
                                output_dims,
                                get_default_value(initial_stats),
                                reverse(input_indices))
        tp_stats = deepcopy(initial_stats)
        def = get_def(tp_stats)
        def.index_order = input.output_indices
        def.level_formats = input.output_formats

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
        formatted_stats = deepcopy(tp_stats)
        def = get_def(formatted_stats)
        def.level_formats = output_formats
        def.index_order = transposed_index_order
        return input, formatted_stats
    end
    return input, initial_stats
end

function transpose_kernel(loop_order::Vector{IndexExpr}, kernel::TensorKernel, stats::TensorStats)
    transposed_index_order =  [x for x in reverse(loop_order) if x in kernel.output_indices]
    is_sorted = is_sorted_wrt_index_order(kernel.output_indices, transposed_index_order)
    new_stats = deepcopy(stats)
    if is_sorted
        def = get_def(new_stats)
        def.index_order = kernel.output_indices
        def.level_formats = kernel.output_formats
        return kernel, new_stats
    end

    input_indices = kernel.output_indices
    input_dict = Dict()
    input_dict["t_1"] = kernel
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
    return input, tp_stats
end
