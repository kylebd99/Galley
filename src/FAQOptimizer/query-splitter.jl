MAX_INDEX_OCCURENCES = 8
# This function takes in queries of the form:
#   Query(name, Aggregate(agg_op, idxs..., map_expr))
# It outputs a set of queries where the final one binds `name` and each
# query has less than `MAX_INDEX_OCCURENCES` index occurences
function split_query(q::PlanNode)

end

function split_queries(p::PlanNode)

end
