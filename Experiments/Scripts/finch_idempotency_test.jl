include("../../src/Galley.jl")

function test_idempotency()
    X = uniform_fiber([1000, 1000], .5)
    X_tag = tag_instance(variable_instance(:X), X)
    X_access = access_instance(X_tag, literal_instance(Reader()), index_instance(:i), index_instance(:j))
    output_fiber = Tensor(SparseHashLevel{1}(Element(0.0)))
    output_tag = tag_instance(variable_instance(:output_fiber), output_fiber)
    output_access = access_instance(output_tag, literal_instance(Updater()), index_instance(:i))
    loop_index_instances = [index_instance(:i), index_instance(:j)]

    full_prgm = assign_instance(output_access, max, X_access)
    for index in loop_index_instances
        full_prgm = loop_instance(index, Dimensionless(), full_prgm)
    end
    initializer = declare_instance(variable_instance(:output_fiber), literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)

    println(typeof(full_prgm))
    println("Type of PROGRAM: ")
    display(Finch.virtualize(:root, typeof(full_prgm), Finch.JuliaContext()))
    println("Type of Tensor: ", typeof(X))
    println("Size of Tensor: ", countstored(X))
    output_fiber = Finch.execute(full_prgm).output_fiber
    println("Input Size: ", countstored(X))
    println("Output Size: ", countstored(output_fiber))
end

test_idempotency()
