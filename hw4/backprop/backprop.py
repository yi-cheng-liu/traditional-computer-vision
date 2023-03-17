import argparse
import os
import pickle
import random
from inspect import signature

from functions import f1, f2, f3_y1, f3_y2, f4


"""
Main entry point for the backprop question.

Students should not need to modify this file.
"""


REQUIRED_FUNCTIONS = [
    ("f1", f1, 1e-10),
    ("f2", f2, 1e-8),
    ("f3 (y=1)", f3_y1, 1e-8),
    ("f3 (y=2)", f3_y2, 1e-8),
]

EXTRA_FUNCTIONS = [
    ("f4", f4, 1e-8),
]

ALL_FUNCTIONS = REQUIRED_FUNCTIONS + EXTRA_FUNCTIONS


parser = argparse.ArgumentParser()
parser.add_argument(
    '--action',
    default='check-both',
    help='What action to perform',
    choices=['check-both', 'check-outputs', 'check-grads', 'save-outputs'])
parser.add_argument(
    '--function',
    default='all',
    help='Which functions to perform the action for',
    choices=['all', 'f1', 'f2', 'f3'])
parser.add_argument(
    '--pkl-path',
    help='Path to a .pkl file with input / output data',
    default='backprop-data.pkl')
parser.add_argument(
    '--seed',
    help='Random seed to use',
    type=int,
    default=442)


def main(args):
    random.seed(args.seed)
    functions = None
    if args.function != 'all':
        functions = [f for f in ALL_FUNCTIONS
                     if f[0].startswith(args.function)]

    if args.action == 'save-outputs':
        fns = ALL_FUNCTIONS if functions is None else functions
        save_outputs(fns, args.pkl_path)
        return

    if args.action in ['check-outputs', 'check-both']:
        fns = REQUIRED_FUNCTIONS if functions is None else functions
        check_outputs(fns, args.pkl_path)
    if args.action in ['check-grads', 'check-both']:
        fns = ALL_FUNCTIONS if functions is None else functions
        check_grads(fns)


def save_outputs(functions, filename, samples_per_fn=100):
    if os.path.isfile(filename):
        msg = f"Output file {filename} already exists, quitting"
        raise ValueError(msg)
    data = {}
    for name, f, _ in functions:
        print(f"Generating data for function {name}")
        num_inputs = len(signature(f).parameters)
        inputs, outputs = [], []
        for _ in range(samples_per_fn):
            xs = tuple(random.random() for _ in range(num_inputs))
            y, _grads = f(*xs)
            inputs.append(xs)
            outputs.append(y)
        data[name] = {'inputs': inputs, 'outputs': outputs}

    print(f"Saving output data to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def check_outputs(functions, filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    all_passed = True
    for name, f, tol in functions:
        print(f"Checking outputs for function {name}")
        if name not in data:
            msg = f"Data for function {name} not found in file {filename}"
            raise ValueError(msg)
        inputs = data[name]['inputs']
        outputs = data[name]['outputs']
        assert len(inputs) == len(outputs)
        passed = True
        for xs, y in zip(inputs, outputs):
            y_pred, _grads = f(*xs)
            if y_pred is None:
                print('  Forward pass not implemented')
                print('  Output check failed!')
                passed = False
                break
            diff = abs(y - y_pred)
            if diff > tol:
                print(f'  Inputs: {xs}')
                print(f'  Expected output: {y}')
                print(f'  Actual output  : {y_pred}')
                print(f'  Difference: {diff}')
                print(f'  Output check failed!')
                passed = False
                break
        if passed:
            print("  Output check passed!")
        else:
            all_passed = False

    if all_passed:
        print("All output checks passed!")
        return True
    else:
        print("At least one output check failed")
    print()
    return False


def numeric_gradient(f, xs, h=1e-4):
    """
    Compute a numeric gradient for a function at a point

    Inputs:
    - f: The function for which to compute a numeric gradient. It should have
      the signature:

      y, grads = f(x1, x2, ..., xN)

      where y and all x are floats.
    - xs: A tuple of length N of floats giving the point at which to compute
      the gradient of f.
    - h: Float giving the step size to use when computing the gradient

    Returns:
    A tuple grads of length N, where grads[i] is the numeric approximation of
    the gradient of y with respect to xs[i].
    """
    grads = []
    xs = list(xs)
    for i, x in enumerate(xs):
        xs[i] = x - h
        y1, _ = f(*xs)
        if y1 is None:
            raise ValueError("Forward pass not implemented")
        xs[i] = x + h
        y2, _ = f(*xs)
        xs[i] = x
        g = (y2 - y1) / (2.0 * h)
        grads.append(g)
    return tuple(grads)


def gradcheck(f, tolerance=1e-10, trials=100000):
    num_inputs = len(signature(f).parameters)
    for _ in range(trials):
        xs = tuple(random.random() for _ in range(num_inputs))
        L, grads = f(*xs)
        try:
            numeric_grads = numeric_gradient(f, xs)
        except ValueError as e:
            print(f'  {e}')
            return False
        max_diff = 0.0
        for g1, g2 in zip(grads, numeric_grads):
            if g1 is None:
                print(f"  Got analytic gradient {grads}")
                print(f"  Backward pass not implemented")
                return False
            diff = abs(g1 - g2)
            max_diff = max(diff, max_diff)
        if max_diff > tolerance:
            print(f"  Analytic gradient: {grads}")
            print(f"  Numeric gradient:  {numeric_grads}")
            print(f"  Max difference: {max_diff}")
            print(f"  Tolerance: {tolerance}")
            return False
    return True


def check_grads(functions):
    all_passed = True
    for name, f, tolerance in functions:
        if len(signature(f).parameters) == 0:
            continue
        print(f"Running gradcheck for {name}")
        passed = gradcheck(f, tolerance)
        if passed:
            print("  Gradcheck passed!")
        else:
            all_passed = False
            print("  Gradcheck failed!")
    if all_passed:
        print("All gradchecks passed!")
        return True
    else:
        print("At least one gradcheck failed")
    return False


if __name__ == '__main__':
    main(parser.parse_args())
