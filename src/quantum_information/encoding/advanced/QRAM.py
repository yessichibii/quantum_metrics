from qiskit import QuantumCircuit, QuantumRegister


class QRAM(QuantumCircuit):
    def __init__(self, n_qbts_position, n_qbts_data):
        super().__init__()
        self.name = "QRAM"
        self.positions = QuantumRegister(n_qbts_position, name="x")
        self.data = QuantumRegister(n_qbts_data, name="f(x)")

        self.add_register(self.positions)
        self.add_register(self.data)

    def initialize_positions(self):
        self.h(self.positions)

    def insertData(self, position, data):
        raise NotImplementedError

