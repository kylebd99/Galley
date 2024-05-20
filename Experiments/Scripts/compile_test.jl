using Finch

println("----------Walk------------")

n = 1000
A_1 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_2 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_3 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_4 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_5 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_6 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
C = Scalar(0.0)

t_1 = @elapsed @finch (for i=_ begin C[] += A_1[i] end end)
t_2 = @elapsed @finch (for i=_ begin C[] += A_1[i] * A_2[i] end end)
t_3 = @elapsed @finch (for i=_ begin C[] += A_1[i] * A_2[i] * A_3[i] end end)
t_4 = @elapsed @finch (for i=_ begin C[] += A_1[i] * A_2[i] * A_3[i] * A_4[i] end end)
t_5 = @elapsed @finch (for i=_ begin C[] += A_1[i] * A_2[i] * A_3[i] * A_4[i] * A_5[i] end end)
t_6 = @elapsed @finch (for i=_ begin C[] += A_1[i] * A_2[i] * A_3[i] * A_4[i] * A_5[i] * A_6[i] end end)

println("t_1: ", t_1)
println("t_2: ", t_2)
println("t_3: ", t_3)
println("t_4: ", t_4)
println("t_5: ", t_5)
println("t_6: ", t_6)

println("----------Gallop------------")

n = 1000
A_1 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_2 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_3 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_4 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_5 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
A_6 = Tensor(SparseList(Element(0.0)), fsprand(n, .001))
C = Scalar(0.0)

t_1 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] end end)
t_2 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] * A_2[gallop(i)] end end)
t_3 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] * A_2[gallop(i)] * A_3[gallop(i)] end end)
t_4 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] * A_2[gallop(i)] * A_3[gallop(i)] * A_4[gallop(i)] end end)
t_5 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] * A_2[gallop(i)] * A_3[gallop(i)] * A_4[gallop(i)] * A_5[gallop(i)] end end)
t_6 = @elapsed @finch (for i=_ begin C[] += A_1[gallop(i)] * A_2[gallop(i)] * A_3[gallop(i)] * A_4[gallop(i)] * A_5[gallop(i)] * A_6[gallop(i)] end end)

println("t_1: ", t_1)
println("t_2: ", t_2)
println("t_3: ", t_3)
println("t_4: ", t_4)
println("t_5: ", t_5)
println("t_6: ", t_6)


println("----------Follow------------")

n = 1000
A_1 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
A_2 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
A_3 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
A_4 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
A_5 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
A_6 = Tensor(Dense(Element(0.0)), fsprand(n, .001))
C = Scalar(0.0)

t_1 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] end end)
t_2 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] * A_2[follow(i)] end end)
t_3 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] * A_2[follow(i)] * A_3[follow(i)] end end)
t_4 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] * A_2[follow(i)] * A_3[follow(i)] * A_4[follow(i)] end end)
t_5 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] * A_2[follow(i)] * A_3[follow(i)] * A_4[follow(i)] * A_5[follow(i)] end end)
t_6 = @elapsed @finch (for i=_ begin C[] += A_1[follow(i)] * A_2[follow(i)] * A_3[follow(i)] * A_4[follow(i)] * A_5[follow(i)] * A_6[follow(i)] end end)

println("t_1: ", t_1)
println("t_2: ", t_2)
println("t_3: ", t_3)
println("t_4: ", t_4)
println("t_5: ", t_5)
println("t_6: ", t_6)
