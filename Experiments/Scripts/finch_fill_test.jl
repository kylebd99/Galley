using Finch
using Finch.FinchNotation: index_instance, tag_instance, variable_instance, literal_instance,
                            Reader, Updater, access_instance, assign_instance, loop_instance,
                            Dimensionless, declare_instance, block_instance

function test_fill()
    X = fsprand((1000, 1000), .5)
    indices = [index_instance(:i), index_instance(:j)]
    X_tag = tag_instance(variable_instance(:X), X)
    X_access = access_instance(X_tag, literal_instance(Reader()), indices...)
    X_instance = X_access
    output_fiber = Tensor(SparseHashLevel{1}(SparseHashLevel{1}(Element(0.0))))
    output_tag = tag_instance(variable_instance(:output_fiber), output_fiber)
    output_access = access_instance(output_tag, literal_instance(Updater()), indices...)
    full_prgm = assign_instance(output_access, literal_instance(initwrite(0.0)), X_instance)

    for index in indices
        full_prgm = loop_instance(index, Dimensionless(), full_prgm)
    end

    initializer = declare_instance(tag_instance(variable_instance(:output_fiber), output_fiber), literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)

    println("Type of PROGRAM: ")
    display(Finch.virtualize(:root, typeof(full_prgm), Finch.JuliaContext()))
    output_fiber = Finch.execute(full_prgm).output_fiber
    println("Input Size: ", countstored(X))
    println("Output Size: ", countstored(output_fiber))
    return output_fiber
end

test_fill()
