# This optimizer returns a single-bag decomposition. It is insufficient for any but the
# smallest faq problems.

function naive_decomposition(faq::FAQInstance)
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    output_index_order = faq.output_index_order
    factors = faq.factors
    bag::Bag = Bag(mult_op, sum_op, factors, faq.input_indices, faq.output_indices, Set{Bag}(), 0)
    return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag, output_index_order)
end
