""" Understands the graph and associates right nodes to right PE."""

import simpy

from fate.DUT.queue import SimpleQueue

class Scheduler:

    def __init__(self, env, parameters) -> None:
        self.env = env
        self.params = parameters
        self.graph = None
        # Use this to maintain the nodes that can be scheduled (all parent nodes have been scheduled)
        self.ready_nodes = SimpleQueue("Ready Nodes")
        self.ready_nodes_container = None
        self.ready_nodes_resource = {}
        self.update_nodes_lock = simpy.Resource(self.env, capacity=1)
        # Use this to maintain the workers executing a particular node
        self.unexecuted_nodes = [] 
        self.executing_nodes = {}

    def update_graph(self, graph):
        """Updates the graph data-structure"""
        self.graph = graph
        self.ready_nodes_container = simpy.Container(self.env, init=1)
        self.unexecuted_nodes = [node.name for node in graph.graph]
        yield self.env.process(self.update_ready_nodes_resource([graph.graph[0].name,]))

    def update_ready_nodes_resource(self, nodes):
        """Updates ready nodes based on avaiability."""        
        for node in nodes:
            self.ready_nodes.push(node)
            self.unexecuted_nodes.remove(node)
            resource = simpy.Resource(self.env, capacity=1)
            req = resource.request()
            yield req
            self.ready_nodes_resource[node] = (resource, req)
            
    def get_node(self, worker_name):
        """
            Returns a node whose dependencies have been satisfied.
            Also updates the ready nodes to include its children.
        """
        # If nothing left, then exit.
        if(self.check_completion()):
            yield self.env.timeout(1)
            return None

        # Get the node id and the node
        node_id = self.ready_nodes.pop()
        node = self.graph.graph[node_id]

        # Update executing nodes
        self.executing_nodes[node_id] = worker_name

        # Update ready nodes
        with self.update_nodes_lock.request() as req:
            yield req 
            if(len(node.children)>0):
                children = list(set([child for child in node.children if((child not in self.ready_nodes.queue) and (child not in self.executing_nodes))]))
                yield self.env.process(self.update_ready_nodes_resource(children))
                if(len(children) > 0):
                    yield self.ready_nodes_container.put(len(children))

        # Release the lock on assigned node
        resource, req = self.ready_nodes_resource[node_id]
        yield resource.release(req)
        # self.print_status()
        return node

    def update_completion(self, node_id):
        """Updates the graph based on which node finished execution"""
        # Remove from executing list
        del self.executing_nodes[node_id]
        del self.ready_nodes_resource[node_id]

    def get_consumer(self, node_id):
        """Return the assigned consumer of any node"""
        children = self.graph.graph[node_id].children
        # Terminal Node
        if(len(children) == 0):
            yield self.env.timeout(1)
            return None
        else:
            consumers = []
            for child in children:
                with self.ready_nodes_resource[child][0].request() as req:
                    yield req
                consumers.append(self.executing_nodes[child])
            return consumers

    def check_completion(self):
        """Check if the program is complete"""
        return (len(self.unexecuted_nodes)==0) and self.ready_nodes.empty()

    def print_status(self):
        """Print"""
        print("\n--------------- Status-----------------")
        print("Ready Nodes: {0} (Credits: {1})".format(self.ready_nodes.queue, self.ready_nodes_container.level))
        print("Executing Nodes: ", self.executing_nodes)
        print("Unexecuted Nodes: ", self.unexecuted_nodes)
        print("----------------------------------------\n")

if __name__ == "__main__":

    print("Running Unit Test for Scheduler")
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

    env = simpy.Environment()
    params = Parameters()
    scheduler = Scheduler(env, params, testGraph)

    def get_node(worker):
        node = yield env.process(scheduler.get_node(worker))
        print("Assigned Node {0} to {1}".format(node.name, worker))
        return node

    def get_consumer_verify(node_id, result):
        consumer = yield env.process(scheduler.get_consumer(node_id))
        assert set(consumer) == set(result), "Consumer Test Failed. Expected {0}, got {1}".format(result, consumer)
        print("\n\n\tConsumer Test Passed.\n\n")

    def completion(node_id):
        yield env.process(scheduler.update_completion(node_id))

    def test_consumer():
        yield env.process(scheduler.update_graph(testGraph))
        scheduler.print_status()
        yield env.process(get_node('Worker1'))
        proc1 = env.process(get_consumer_verify(0, ['Worker3','Worker2']))
        proc3 = env.process(get_node('Worker2'))
        proc4 = env.process(get_node('Worker3'))
        yield env.all_of([proc1, proc3, proc4])

    proc = env.process(test_consumer())
    env.run(proc)
