# This file performs the actual execution of an ExtendedFreeJoin plan. 
using DataStructures
include("PhysicalQueryPlan.jl")


function initialize_tensor(formats::Vector{LevelFormat}, dims::Vector{Int64}, default_value)
    B = Element(default_value)
    for i in range(1, length(dims))
        if formats[i] == t_sparse_list
            B = SparseList(B, dims[i])
            println(B)
        elseif formats[i] == t_dense
            B = Dense(B, dims[i])
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
        if typeof(cur_node) == OperatorExp
            for child_node in cur_node.inputs
                node_id_counter += 1
                enqueue!(nodes_to_visit, (child_node, node_id_counter))
                push!(child_node_ids, node_id_counter)
            end
        elseif typeof(cur_node) == AggregateExp
            node_id_counter += 1
            enqueue!(nodes_to_visit, (cur_node.input, node_id_counter))
            push!(child_node_ids, node_id_counter)
        end
        node_dict[cur_node_id] = (cur_node, child_node_ids)
    end

    default_value = nothing
    for node_id in reverse(range(0, length(keys(node_dict))-1))
        node, child_node_ids = node_dict[node_id]
        if typeof(node) == TensorAccess
            node_dict[node_id] = Finch.default(kernel.input_tensors[node.tensor_id])
        elseif typeof(node) == OperatorExp
            child_vals = [node_dict[x] for x in child_node_ids]
            node_dict[node_id] = node.op(child_vals...)
        end
        if node_id == 0
            if typeof(node) == AggregateExp
                default_value = node_dict[child_node_ids[1]]
            else
                default_value = node_dict[node_id]
            end
        elseif typeof(node) == AggregateExp
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
            protocol = fastwalk
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


function execute_tensor_kernel(kernel::TensorKernel)
    loop_order = [Finch.FinchNotation.index_instance(Symbol(i)) for i in kernel.loop_order]
    output_indices = [Finch.FinchNotation.index_instance(Symbol(i)) for i in kernel.output_indices]
    output_dimensions = Vector{Int64}()
    for index in kernel.output_indices
        for tensor_id in keys(kernel.input_indices)
            input_dim_number = findfirst(==(index), kernel.input_indices[tensor_id])
            if isa(input_dim_number, Int64)
                push!(output_dimensions, size(kernel.input_tensors[tensor_id])[input_dim_number])
                break
            end
        end
    end
    output_tensor = initialize_tensor(kernel.output_formats, output_dimensions, determine_default_value(kernel))

    tensor_accesses = Dict()
    for tensor_id in keys(kernel.input_tensors)
        tensor_accesses[tensor_id] = initialize_access(tensor_id, kernel.input_tensors[tensor_id], kernel.input_indices[tensor_id], kernel.input_protocols[tensor_id])
    end

    nodes_to_visit = Queue{Tuple{TensorExpression, Int64}}()
    node_dict = Dict()
    node_id_counter = 0
    enqueue!(nodes_to_visit, (kernel.kernel_root, node_id_counter))
    while length(nodes_to_visit) > 0
        cur_node, cur_node_id = dequeue!(nodes_to_visit)
        child_node_ids = []
        if typeof(cur_node) == OperatorExp
            for child_node in cur_node.inputs
                node_id_counter += 1
                enqueue!(nodes_to_visit, (child_node, node_id_counter))
                push!(child_node_ids, node_id_counter)
            end
        elseif typeof(cur_node) == AggregateExp
            node_id_counter += 1
            enqueue!(nodes_to_visit, (cur_node.input, node_id_counter))
            push!(child_node_ids, node_id_counter)
        end
        node_dict[cur_node_id] = (cur_node, child_node_ids)
    end

    agg_op = nothing
    kernel_prgm = nothing
    for node_id in reverse(range(0, length(keys(node_dict))-1))
        node, child_node_ids = node_dict[node_id]
        if typeof(node) == TensorAccess
            node_dict[node_id] = tensor_accesses[node.tensor_id]
        elseif typeof(node) == OperatorExp
            child_prgms = [node_dict[x] for x in child_node_ids]
            node_dict[node_id] = @finch_program_instance $(node.op)(child_prgms...)
        end
        if node_id == 0
            if typeof(node) == AggregateExp
                kernel_prgm = node_dict[child_node_ids[1]]
                agg_op = node.op
            else
                kernel_prgm = node_dict[node_id]
            end
        elseif typeof(node) == AggregateExp
            throw(ArgumentError("Cannot have an aggregate in the middle of a tensor kernel. They must always occur as the outermost operator."))
        end
    end
    if agg_op === nothing
        full_prgm = @finch_program_instance @loop loop_order... output_tensor[output_indices...] = $kernel_prgm
    else
        full_prgm = @finch_program_instance @loop loop_order... output_tensor[output_indices...] <<agg_op>>=  $kernel_prgm
    end
    return Finch.execute(full_prgm).output_tensor
end

