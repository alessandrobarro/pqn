import numpy as np
import torch

class OPGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.E = vertices * (vertices - 1) // 2  # Number of edges in a fully connected graph
        self.prizes = np.random.randint(0, 16, size=self.V)
        self.costs = np.random.randint(1, 11, size=self.V)  # Vertex costs that will translate into edge costs with get_cost()xxs
        self.B = (self.V / 4) * (np.mean(self.prizes) - np.mean(self.costs))

    def get_cost(self, v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        return abs(self.costs[v1] - self.costs[v2])

class OPEnvironment:
    def __init__(self, graph_size):
        self.graph = OPGraph(graph_size)
        self.current_vertex = 0  # Start vertex
        self.route = [self.current_vertex]
        self.budget_spent = 0

    def _get_state_representation(self):
        # R: The current route
        R = self.route + [-1] * (self.graph.V - len(self.route))  # Pad with -1 to ensure consistent length
        
        # X: Indicator vector for visited vertices
        X = [1 if i in self.route else 0 for i in range(self.graph.V)]
        
        # B: Remaining budget
        B = self.graph.B - self.budget_spent
        
        return R + X + [B]
        
    def reset(self):
        self.current_vertex = 0  # start from vertex 0
        self.route = [self.current_vertex]
        self.budget_spent = 0
        
        self.state = self._get_state_representation()
        return self.state
    
    def step(self, action):
        next_vertex = action

        reward = self.graph.prizes[next_vertex] - self.graph.get_cost(self.current_vertex, next_vertex)
        
        if next_vertex in self.route:
            reward = -10  # penalize for revisiting a vertex
        else:
            reward = self.graph.prizes[next_vertex] - self.graph.get_cost(self.current_vertex, next_vertex)
            # Updating budget
            self.budget_spent += self.graph.get_cost(self.current_vertex, next_vertex)
            # Updating current vertex and route
            self.current_vertex = next_vertex
            self.route.append(next_vertex)

        self.state = self._get_state_representation()
        
        # Check termination condition
        done = self.budget_spent > self.graph.B
        
        return self.state, reward, done, {}
