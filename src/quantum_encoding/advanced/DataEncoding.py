from qiskit import QuantumCircuit

class DataEncoding(QuantumCircuit):

    def __init__(self, data):
        self.data = data

    def encode():
        """Helper that returns a quantum circuit corresponding to the
        encoded data based on what it's surposed to encode
        """
        raise NotImplementedError
    