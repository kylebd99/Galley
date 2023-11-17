# This optimizer returns a single-bag decomposition. It is insufficient for any but the
# smallest faq problems.

function naive_decomposition(faq::FAQInstance)
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    factors = faq.factors
    bag::Bag = Bag(mult_op, sum_op, factors, faq.input_indices, faq.output_indices, Bag[])
    return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag)
end
