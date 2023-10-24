

function naive_decomposition(faq::FAQInstance)
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    factors = faq.factors
    bag::Bag = Bag(factors, faq.input_indices, faq.output_indices, Bag[])
    return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag)
end

function faq_to_htd(faq::FAQInstance)
    return naive_decomposition(faq)
end
