# This file performs the actual execution of an ExtendedFreeJoin plan. 
using DataStructures
using PrettyPrinting
using Finch: @finch_program_instance, SparseHashLevel
include("PhysicalOptimizer.jl")


function initialize_tensor(formats::Vector{LevelFormat}, dims::Vector{Int64}, default_value)
    B = Element(default_value)
    for i in range(1, length(dims))
        if formats[i] == t_sparse_list
            B = SparseList(B, dims[i])
        elseif formats[i] == t_dense
            B = Dense(B, dims[i])
        elseif formats[i] == t_hash
            B = SparseHashLevel(B, Tuple([dims[i]]))
        else
            println("Error: Attempted to initialize invalid level format type.")
        end
    end
    return Fiber!(B)
end

function determine_default_value(kernel::TensorKernel)
    nodes_to_visit = Queue{Tuple{TensorExpression, Int64}}()
    node_dict = Dict()
    node_id_counter = 0
    enqueue!(nodes_to_visit, (kernel.kernel_root, node_id_counter))
    while length(nodes_to_visit) > 0
        cur_node, cur_node_id = dequeue!(nodes_to_visit)
        child_node_ids = []
        if typeof(cur_node) == OperatorExpr
            for child_node in cur_node.inputs
                node_id_counter += 1
                enqueue!(nodes_to_visit, (child_node, node_id_counter))
                push!(child_node_ids, node_id_counter)
            end
        elseif typeof(cur_node) == AggregateExpr
            node_id_counter += 1
            enqueue!(nodes_to_visit, (cur_node.input, node_id_counter))
            push!(child_node_ids, node_id_counter)
        end
        node_dict[cur_node_id] = (cur_node, child_node_ids)
    end

    default_value = nothing
    for node_id in reverse(range(0, length(keys(node_dict))-1))
        node, child_node_ids = node_dict[node_id]
        if node isa InputExpr
            if kernel.input_tensors[node.tensor_id] isa Number
                node_dict[node_id] = kernel.input_tensors[node.tensor_id]
            else
                node_dict[node_id] = Finch.default(kernel.input_tensors[node.tensor_id])
            end
        elseif node isa OperatorExpr
            child_vals = [node_dict[x] for x in child_node_ids]
            node_dict[node_id] = node.op(child_vals...)
        end
        if node_id == 0
            if node isa AggregateExpr
                default_value = node_dict[child_node_ids[1]]
            else
                default_value = node_dict[node_id]
            end
        elseif node isa AggregateExpr
            throw(ArgumentError("Cannot have an aggregate in the middle of a tensor kernel. They must always occur as the outermost operator."))
        end
    end
    return default_value
end

function initialize_access(tensor_id::TensorId, tensor::Fiber, index_ids::Vector{String}, protocols::Vector{AccessProtocol})
    index_expressions = []
    for i in range(1, length(index_ids))
        index = Finch.FinchNotation.index_instance(Symbol(index_ids[i]))
        protocol = nothing
        if protocols[i] == t_walk
            protocol = walk
        elseif protocols[i] == t_fast_walk
            protocol = laminate
        elseif protocols[i] == t_follow
            protocol = follow
        elseif protocols[i] == t_lead
            protocol = lead
        elseif protocols[i] == t_gallop
            protocol = gallop
        end
        push!(index_expressions, @finch_program_instance index::protocol)
    end
    tensor_var = Finch.FinchNotation.variable_instance(Symbol(tensor_id), tensor)
    return @finch_program_instance $(tensor_var)[index_expressions...]
end


function execute_tensor_kernel(kernel::TensorKernel; lvl = 1, verbose=0)
    verbose >= 2 && println(lvl)
    for tensor_id in keys(kernel.input_tensors)
        if kernel.input_tensors[tensor_id] isa TensorKernel
            kernel.input_tensors[tensor_id] = execute_tensor_kernel(kernel.input_tensors[tensor_id], lvl=lvl+1, verbose=verbose)
        end
    end
    verbose >= 2 && println(lvl)

    nodes_to_visit = Queue{Tuple{TensorExpression, Int64}}()
    node_dict = Dict()
    node_id_counter = 0
    enqueue!(nodes_to_visit, (kernel.kernel_root, node_id_counter))
    while length(nodes_to_visit) > 0
        cur_node, cur_node_id = dequeue!(nodes_to_visit)
        child_node_ids = []
        if typeof(cur_node) == OperatorExpr
            for child_node in cur_node.inputs
                node_id_counter += 1
                enqueue!(nodes_to_visit, (child_node, node_id_counter))
                push!(child_node_ids, node_id_counter)
            end
        elseif typeof(cur_node) == AggregateExpr
            node_id_counter += 1
            enqueue!(nodes_to_visit, (cur_node.input, node_id_counter))
            push!(child_node_ids, node_id_counter)
        end
        node_dict[cur_node_id] = (cur_node, child_node_ids)
    end

    index_dims = Dict()
    agg_op = nothing
    kernel_prgm = nothing
    for node_id in reverse(range(0, length(keys(node_dict))-1))
        node, child_node_ids = node_dict[node_id]
        if node isa InputExpr
            tensor_id = node.tensor_id
            if kernel.input_tensors[tensor_id] isa Number
                node_dict[node_id] = Finch.FinchNotation.literal_instance(kernel.input_tensors[tensor_id])
            else
                for idx in node.input_indices
                    if idx in keys(index_dims) && index_dims[idx] != node.stats.dim_size[idx]
                        throw(ArgumentError("Error: Tensors cannot be joined on indices with different dimension sizes."))
                    end
                    index_dims[idx] = node.stats.dim_size[idx]
                end
                node_dict[node_id] = initialize_access(tensor_id, kernel.input_tensors[tensor_id], node.input_indices, node.input_protocols)
            end
        elseif node isa OperatorExpr
            child_prgms = [node_dict[x] for x in child_node_ids]
            op = node.op
            node_dict[node_id] = @finch_program_instance op(child_prgms...)
        end
        if node_id == 0
            if node isa AggregateExpr
                kernel_prgm = node_dict[child_node_ids[1]]
                agg_op = node.op
            else
                kernel_prgm = node_dict[node_id]
            end
        elseif node isa AggregateExpr
            throw(ArgumentError("Cannot have an aggregate in the middle of a tensor kernel. They must always occur as the outermost operator."))
        end
    end

    loop_order = [Finch.FinchNotation.index_instance(Symbol(i)) for i in kernel.loop_order]
    output_indices = [Finch.FinchNotation.index_instance(Symbol(i)) for i in kernel.output_indices]
    output_dimensions = Vector{Int64}()
    for idx in kernel.output_indices
        push!(output_dimensions, index_dims[idx])
               
    end
    output_default = determine_default_value(kernel)
    output_tensor = initialize_tensor(kernel.output_formats, output_dimensions, output_default)

    if agg_op === nothing
        default_value = Finch.FinchNotation.literal_instance(output_default)
        full_prgm = @finch_program_instance (output_tensor .= $default_value; @loop loop_order... output_tensor[output_indices...] = $kernel_prgm)
    else
        default_value = Finch.FinchNotation.literal_instance(output_default)
        full_prgm = @finch_program_instance (output_tensor .= $default_value; @loop loop_order... output_tensor[output_indices...] <<agg_op>>=  $kernel_prgm)
    end

    for tensor_id in keys(kernel.input_tensors)
        if kernel.input_tensors[tensor_id] isa TensorKernel
            kernel.input_tensors[tensor_id] = nothing
        end
    end

    output_tensor = Finch.execute(full_prgm).output_tensor
    verbose >= 2 && println("Kernel: ", kernel.kernel_root)
    if verbose >= 1
        println("Expected Output Tensor Size: ", kernel.stats.cardinality)
        println("Output Tensor Size: ", countstored(output_tensor))
    end
    return output_tensor
end

