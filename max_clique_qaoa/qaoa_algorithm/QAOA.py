from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import ComputationalBasisState
from quri_parts.algo.optimizer import Adam
from quri_algo.circuit.time_evolution import TrotterTimeEvolutionCircuitFactory
from quri_algo.circuit.time_evolution.exact_unitary import ExactUnitaryTimeEvolutionCircuitFactory
import networkx as nx
import numpy as np

def get_hamiltonian_from_QUBO(Q):
    # Convert QUBO to Hamiltonian operator
    hamiltonian = Operator()
    for i in range(n):
        for j in range(i, n):
            if abs(Q[i,j]) > 1e-10:
                if i == j:
                    # Diagonal terms
                    hamiltonian.add_term(pauli_label(f"Z{i}"), Q[i,i])
                else:
                    # Off-diagonal terms
                    hamiltonian.add_term(pauli_label(f"Z{i}Z{j}"),Q[i,j])

    return hamiltonian

def get_QUBO_from_graph(G):
    # Convert max clique to QUBO
    n = len(G.nodes())
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if not G.has_edge(i,j):
                    Q[i,j] += 1
                Q[i,i] += -1
                Q[j,j] += -1

    return Q

def main():
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,0), (0,2)])  # Sample graph
    hamiltonian = get_hamiltonian_from_QUBO(get_QUBO_from_graph(G))



    # Setup QAOA
    depth = 2  # QAOA depth parameter
    ansatz = Symm(hamiltonian, depth)
    circuit = LinearMappedUnboundParametricQuantumCircuit(n)
    circuit += ansatz

    # Initial parameters
    init_params = np.random.rand(2 * depth)

    # Optimizer
    optimizer = Adam()
    optimizer.set_parameters(init_params)

    # Run optimization (Note: This is a simplified version, actual optimization would need a quantum backend)
    steps = 100
    for step in range(steps):
        # Run circuit with current parameters
        parameterized_circuit = circuit.bind_parameters(optimizer.parameters)
        
        # Initialize state in equal superposition
        init_state = ComputationalBasisState(n, bits=0)
        
        # Calculate expectation value
        # Note: In practice this would be done on quantum hardware
        # Here we're using a simplified calculation
        expectation = 0.0
        
        # Calculate gradient
        grad = np.zeros_like(optimizer.parameters)
        # Calculate gradient using parameter shift rule
        # For QAOA, the shift amount is pi/2 for gamma (even indices) and pi/4 for beta (odd indices)
        for i in range(len(optimizer.parameters)):
            shift = np.pi/2 if i % 2 == 0 else np.pi/4
            
            params_plus = optimizer.parameters.copy()
            params_minus = optimizer.parameters.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            # Calculate expectation values at shifted parameters
            # In practice these would be quantum measurements
            exp_plus = 0.0  # expectation with positive shift
            exp_minus = 0.0 # expectation with negative shift
            
            grad[i] = (exp_plus - exp_minus) / 2
        
        # Update parameters using optimizer
        optimizer.update_parameters(-grad)
        
        if step % 10 == 0:
            print(f"Step {step}, Energy: {expectation:.4f}")

    print("QAOA optimization completed")
    print("Final parameters:", optimizer.parameters)
