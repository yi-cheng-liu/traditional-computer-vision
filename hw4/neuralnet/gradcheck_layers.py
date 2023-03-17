import numpy as np
from utils import numeric_backward, numeric_gradient
from layers import fc_forward, fc_backward
from layers import relu_forward, relu_backward
from layers import softmax_loss, l2_regularization

def gradcheck_fc():
    return gradcheck_fc_helper(fc_forward, fc_backward)

def gradcheck_relu():
    return gradcheck_relu_helper(relu_forward, relu_backward)

def gradcheck_softmax():
    return gradcheck_softmax_helper(softmax_loss)

def gradcheck_l2_regularization():
    return gradcheck_l2_regularization_helper(l2_regularization)

def gradcheck_fc_helper(forward, backward):
    print('Running numeric gradient check for fc')
    N, Din, Dout = 3, 4, 5
    x = np.random.randn(N, Din)
    w = np.random.randn(Din, Dout)
    b = np.random.randn(Dout)

    y, cache = forward(x, w, b)
    if y is None:
        print('  Forward pass is not implemented!')
        return None, None, None

    grad_y = np.random.randn(*y.shape)
    grad_x, grad_w, grad_b = backward(grad_y, cache)
    if grad_x is None or grad_w is None or grad_b is None:
        print('  Backward pass is not implemented!')
        return None, None, None

    fx = lambda _: forward(_, w, b)[0]
    grad_x_numeric = numeric_backward(fx, x, grad_y)
    grad_x_diff = np.abs(grad_x - grad_x_numeric).max()
    print('  grad_x difference: ', grad_x_diff)

    fw = lambda _: forward(x, _, b)[0]
    grad_w_numeric = numeric_backward(fw, w, grad_y)
    grad_w_diff = np.abs(grad_w - grad_w_numeric).max()
    print('  grad_w difference: ', grad_w_diff)

    fb = lambda _: forward(x, w, _)[0]
    grad_b_numeric = numeric_backward(fb, b, grad_y)
    grad_b_diff = np.abs(grad_b - grad_b_numeric).max()
    print('  grad_b difference: ', grad_b_diff)
    return grad_x_diff, grad_w_diff, grad_b_diff


def gradcheck_relu_helper(forward, backward):
    print('Running numeric gradient check for relu')
    N, Din = 4, 5
    x = np.random.randn(N, Din)

    y, cache = forward(x)
    if y is None:
        print('  Forward pass is not implemented!')
        return None

    grad_y = np.random.randn(*y.shape)
    grad_x = backward(grad_y, cache)
    if grad_x is None:
        print('  Backward pass is not implemented!')
        return None

    f = lambda _: forward(_)[0]
    grad_x_numeric = numeric_backward(f, x, grad_y)
    grad_x_diff = np.abs(grad_x - grad_x_numeric).max()
    print('  grad_x difference: ', grad_x_diff)
    return grad_x_diff

def gradcheck_softmax_helper(fn):
    print('Running numeric gradient check for softmax loss')
    N, C = 4, 5
    x = np.random.randn(N, C)
    y = np.random.randint(C, size=(N,))
    loss, grad_x = fn(x, y)
    if loss is None or grad_x is None:
        print('  Softmax not implemented!')
        return None

    f = lambda _: fn(_, y)[0]
    grad_x_numeric = numeric_gradient(f, x)
    grad_x_diff = np.abs(grad_x - grad_x_numeric).max()
    print('  grad_x difference: ', grad_x_diff)
    return grad_x_diff

def gradcheck_l2_regularization_helper(fn):
    print('Running numeric gradient check for L2 regularization')
    Din, Dout = 3, 4
    reg = 0.1
    w = np.random.randn(Din, Dout)
    loss, grad_w = fn(w, reg)
    if loss is None or grad_w is None:
        print('  L2 regularization not implemented!')
        return None

    f = lambda _: fn(_, reg)[0]
    grad_w_numeric = numeric_gradient(f, w)
    grad_x_diff = np.abs(grad_w - grad_w_numeric).max()
    print('  grad_w difference: ', grad_x_diff)
    return grad_x_diff

def main():
    gradcheck_fc()
    gradcheck_relu()
    gradcheck_softmax()
    gradcheck_l2_regularization()


if __name__ == '__main__':
    main()
