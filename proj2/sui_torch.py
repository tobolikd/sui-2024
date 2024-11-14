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
            self.grad = deltas
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.grad = np.array(1.0)

        if self.back_op:
            self.back_op(self)

def sui_sum(tensor):
    raise NotImplementedError()

def add(a, b):
    raise NotImplementedError()


def subtract(a, b):
    raise NotImplementedError()


### multiply
def multiply(a, b):
    back_op = lambda result: multiply_backward(result, a, b)
    return Tensor(a.value * b.value, back_op=back_op)

def multiply_backward(result, a, b):
    raise NotImplementedError()

### relu
def relu(tensor):
    return Tensor(np.maximum(0, tensor.value), back_op=relu_backward)

def relu_backward(tensor):
    tensor.grad *= tensor.value > 0

### dot
def dot_product(a, b):
    back_op = lambda result: dot_product_backward(result, a, b)
    return Tensor(np.dot(a.value, b.value), back_op=back_op)

def dot_product_backward(result, a, b):
    b_T = np.transpose(b.value)
    a_T = np.transpose(a.value)
    a.grad += np.dot(result.grad, b_T)
    b.grad += np.dot(a_T, result.grad)
