import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = "\t" + "\n\t".join(str_val.split("\n"))
        str_bwd = str(self.back_op.__class__.__name__)
        return "Tensor(\n" + str_val + "\n\tbwd: " + str_bwd + "\n)"

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert (
                deltas.shape == self.value.shape
            ), f"Expected gradient with shape {self.value.shape}, got {deltas.shape}"
            self.grad = deltas
            if self.back_op:
                self.back_op()
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(
                    f"Can only backpropagate a scalar, got shape {self.shape}"
                )

            if self.back_op is None:
                raise ValueError(f"Cannot start backpropagation from a leaf!")

            self.grad = np.array(1.0)
            self.back_op()


###sum
def sui_sum(tensor):
    value = np.sum(tensor.value)
    result = Tensor(value)

    def back_op():
        tensor.grad += np.ones_like(tensor.value) * result.grad
        tensor.backward(tensor.grad)

    result.back_op = back_op
    return result


###add
def add(a, b):
    if a.value.shape != b.value.shape:
        raise ValueError("Tensor sizes must match for elementwise addition.")

    value = a.value + b.value
    result = Tensor(value)

    def back_op():
        a.grad += result.grad
        b.grad += result.grad

        a.backward(result.grad)
        b.backward(result.grad)

    result.back_op = back_op
    return result


###substract
def subtract(a, b):
    if a.value.shape != b.value.shape:
        raise ValueError("Tensor sizes must match for elementwise addition.")

    value = a.value - b.value
    result = Tensor(value)

    def back_op():
        a.grad += result.grad
        b.grad -= result.grad

        a.backward(result.grad)
        b.backward(-result.grad)

    result.back_op = back_op
    return result


### multiply
def multiply(a, b):
    result = Tensor(a.value * b.value)

    def back_op():
        a.grad += b.value * result.grad
        b.grad += a.value * result.grad
        a.backward(a.grad)
        b.backward(b.grad)

    result.back_op = back_op
    return result


### relu
def relu(tensor):
    result = Tensor(np.maximum(0, tensor.value))

    def back_op():
        tensor.grad += result.grad * (tensor.value > 0)
        tensor.backward(tensor.grad)

    result.back_op = back_op
    return result


### dot
def dot_product(a, b):
    result = Tensor(np.dot(a.value, b.value))

    def back_op():
        b_T = np.transpose(b.value)
        a_T = np.transpose(a.value)
        a.grad += np.dot(result.grad, b_T)
        b.grad += np.dot(a_T, result.grad)
        a.backward(a.grad)
        b.backward(b.grad)

    result.back_op = back_op
    return result
