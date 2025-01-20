from typing import Sequence

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.estimator import QuantumEstimator
from quri_parts.qulacs.estimator import create_qulacs_general_vector_estimator
from quri_parts.core.state import ComputationalBasisState
from quri_parts.algo.optimizer import Adam, OptimizerStatus
from quri_algo.problem.hamiltonian import QubitHamiltonianInput
from quri_algo.circuit.time_evolution.interface import TimeEvolutionCircuitFactory
from quri_algo.circuit.time_evolution import TrotterTimeEvolutionCircuitFactory
from quri_algo.circuit.time_evolution.exact_unitary import ExactUnitaryTimeEvolutionCircuitFactory
import networkx as nx
import numpy as np

def get_mixing_hamiltonian(n_qubits):
    hamiltonian = Operator()
    for i in range(n_qubits):
        hamiltonian.add_term(pauli_label(f"X{i}"),1.0)
    
    return hamiltonian

def get_cost_function(hamiltonian: Operator, time_evo_factories: Sequence[TimeEvolutionCircuitFactory], estimator: QuantumEstimator):
    def cost_fn(params: Sequence[float]):
        qubit_count = time_evo_factories[0].qubit_count
        circuit = QuantumCircuit(qubit_count)
        for p, f in zip(params(), time_evo_factories):
            circuit += f(p)
        state = GeneralCircuitQuantumState(qubit_count, circuit)
        return estimator(hamiltonian, state).value.real
    return cost_fn

def qaoa_exact_time_evo(hamiltonian: Operator, n_qubits: int, n_steps: int):
    isinghamiltonian = QubitHamiltonianInput(n_qubits, hamiltonian)
    mixinghamiltonian = QubitHamiltonianInput(n_qubits, get_mixing_hamiltonian(n_qubits))
    time_evo_factories = []
    for _ in range(n_steps):
        time_evo_factories.append(ExactUnitaryTimeEvolutionCircuitFactory(isinghamiltonian))
        time_evo_factories.append(ExactUnitaryTimeEvolutionCircuitFactory(mixinghamiltonian))
    
    estimator = create_qulacs_general_vector_estimator()
    cost_fn = get_cost_function(hamiltonian,time_evo_factories, estimator)
    rng = np.random.default_rng()
    params = rng.random(n_steps*2)

    optimizer = Adam()
    state = optimizer.get_init_state(params)
    while state.status == OptimizerStatus.SUCCESS:
        state = optimizer.step(state, cost_fn)

    return state

def qaoa_trotter(hamiltonian: Operator, n_qubits: int, n_steps: int, n_trotter: int):
    isinghamiltonian = QubitHamiltonianInput(n_qubits, hamiltonian)
    mixinghamiltonian = QubitHamiltonianInput(n_qubits, get_mixing_hamiltonian(n_qubits))
    time_evo_factories = []
    for _ in range(n_steps):
        time_evo_factories.append(TrotterTimeEvolutionCircuitFactory(isinghamiltonian, n_trotter))
        time_evo_factories.append(TrotterTimeEvolutionCircuitFactory(mixinghamiltonian, n_trotter))
    
    estimator = create_qulacs_general_vector_estimator()
    cost_fn = get_cost_function(hamiltonian,time_evo_factories, estimator)
    rng = np.random.default_rng()
    params = rng.random(n_steps*2)

    optimizer = Adam()
    state = optimizer.get_init_state(params)
    while state.status == OptimizerStatus.SUCCESS:
        state = optimizer.step(state, cost_fn)

    return state
