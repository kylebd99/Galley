# This file is the entrance point to FAQ decompositions. First, it does some simple
# optimizations to make the problem smaller, then it hands it off to an HTD algorithm.

function faq_to_htd(faq::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS = naive, use_validation = true)
    validate_faq(faq)
    htd = nothing
    if faq_optimizer == naive
        htd = return naive_decomposition(faq)
    elseif faq_optimizer == hypertree_width
        htd = return hypertree_width_decomposition(faq)
    elseif faq_optimizer == greedy
        htd = return greedy_decomposition(faq)
    elseif faq_optimizer == ordering
        htd = return order_based_decomposition(faq)
    else
        throw(ArgumentError(string(faq_optimizer) * " is not supported yet."))
    end
    use_validation && validate_htd(faq, htd)
    return htd
end

function validate_faq(faq::FAQInstance)
    all_factors_indices = union([get_plan_node_indices(factor.input) for factor in faq.factors]...)
    @assert faq.input_indices == all_factors_indices
end

function validate_htd(faq::FAQInstance, htd::HyperTreeDecomposition)
    @assert faq.output_indices == htd.output_indices
    @assert faq.output_index_order == htd.output_index_order

    function _validate_indices(factor::Factor)
        input_node_indices = get_plan_node_indices(factor.input)
        @assert factor.all_indices == input_node_indices
        @assert get_index_set(factor.stats) == input_node_indices
        @assert get_index_set(factor.stats) == get_index_set(factor.input.stats)
    end

    function _validate_indices(parent_indices::Set{IndexExpr}, bag::Bag)
        covered_indices = union([get_index_set(factor.stats) for factor in edge_cover]...)
        @assert bag.covered_indices == covered_indices
        for factor in bag.edge_covers
            _validate_indices(factor)
        end
    end
end
