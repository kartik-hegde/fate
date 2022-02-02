from fate.utils.graph import Graph
from fate.parameters import Parameters

# Create a random Graph.
testGraph = Graph()
program  = ['Init', 'GetHandles A', 'GetCoords A', 'GetHandles B', 'GetCoords B', 'Intersect', 'Compute']
# Insert the parent
testGraph.graph[0].payload = program[0]
# Insert all the work
testGraph.add_edges([0],[2],program[1])
testGraph.add_edges([1],[5],program[2])
testGraph.add_edges([0],[4],program[3])
testGraph.add_edges([3],[5],program[4])
testGraph.add_edges([2,4],[6],program[5])
testGraph.add_edges([5],[],program[6])
testGraph.printGraph()