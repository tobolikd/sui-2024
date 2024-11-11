import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'

            raise NotImplementedError('Backpropagation with deltas not implemented yet')
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            raise NotImplementedError('Backpropagation without deltas not implemented yet')


def sui_sum(tensor):
    raise NotImplementedError()


def add(a, b):
    raise NotImplementedError()


def subtract(a, b):
    raise NotImplementedError()


def multiply(a, b):
    raise NotImplementedError()


def relu(tensor):
    raise NotImplementedError()


def dot_product(a, b):
    raise NotImplementedError()
