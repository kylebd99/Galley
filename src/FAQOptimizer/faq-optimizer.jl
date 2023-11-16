# This file is the entrance point to FAQ decompositions. First, it does some simple
# optimizations to make the problem smaller, then it hands it off to an HTD algorithm.

function faq_to_htd(faq::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS = naive)
    prune_faq!(faq)
    if faq_optimizer == naive
        return naive_decomposition(faq)
    elseif faq_optimizer == hypertree_width
        return hypertree_width_decomposition(faq)
    else
        throw(ArgumentError(string(faq_optimizer) * " is not supported yet."))
    end
end
