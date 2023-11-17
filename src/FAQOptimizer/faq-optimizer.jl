# This file is the entrance point to FAQ decompositions. First, it does some simple
# optimizations to make the problem smaller, then it hands it off to an HTD algorithm.

function faq_to_htd(faq::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS = naive)
    if faq_optimizer == naive
        prune_faq!(faq)
        return naive_decomposition(faq)
    elseif faq_optimizer == hypertree_width
        prune_faq!(faq)
        return hypertree_width_decomposition(faq)
    elseif faq_optimizer == greedy
        return greedy_decomposition(faq)
    else
        throw(ArgumentError(string(faq_optimizer) * " is not supported yet."))
    end
end
