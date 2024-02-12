using LightGraphs
using QXGraphDecompositions


g = LightGraphs.Graph(12)
LightGraphs.add_edge!(g, 1, 2)
LightGraphs.add_edge!(g, 1, 3)
LightGraphs.add_edge!(g, 1, 4)
LightGraphs.add_edge!(g, 2, 3)
LightGraphs.add_edge!(g, 2, 4)
LightGraphs.add_edge!(g, 3, 4)

LightGraphs.add_edge!(g, 5, 6)
LightGraphs.add_edge!(g, 5, 7)
LightGraphs.add_edge!(g, 5, 8)
LightGraphs.add_edge!(g, 6, 7)
LightGraphs.add_edge!(g, 6, 8)
LightGraphs.add_edge!(g, 7, 8)

LightGraphs.add_edge!(g, 9, 10)
LightGraphs.add_edge!(g, 9, 11)
LightGraphs.add_edge!(g, 9, 12)
LightGraphs.add_edge!(g, 10, 11)
LightGraphs.add_edge!(g, 10, 12)
LightGraphs.add_edge!(g, 11, 12)

LightGraphs.add_edge!(g, 1, 5)
LightGraphs.add_edge!(g, 6, 9)
LightGraphs.add_edge!(g, 12, 4)

td = flow_cutter(g)
