#!/usr/bin/python
import sys, os, pdb, argparse, traceback  # noqa: E401
import numpy as np
import warmups, tests, common  # noqa: E401


STORAGE = "442solutions/"
SEEDS = [442, 1337, 31415, 3777, 2600]
SEEDS += [s * 2 for s in SEEDS] + [s * 5 for s in SEEDS]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Which to test")
    parser.add_argument("--allwarmups", action="store_true", default=False,
                        help="Run all warmups")
    parser.add_argument("--alltests", action="store_true", default=False,
                        help="Run all tests")
    parser.add_argument("--allconvs", action="store_true", default=False,
                        help="Run all convolution tests")
    parser.add_argument("--store", action="store_true", default=False,
                        help="Overwrite cache of answers")
    parser.add_argument("--pdb", action="store_true", default=False,
                        help="Launch PDB when answers don't match")
    args = parser.parse_args()
    if not (args.allwarmups or args.alltests or args.allconvs):
        if args.test is None:
            print("Need a test to run")
            sys.exit(1)
    return args


def storeMultiRes(testName, resultsDict):
    # Given resultsDict is a dictionary of str(seed): result
    if not os.path.exists(STORAGE):
        os.mkdir(STORAGE)
    resName = "%s/%s.npz" % (STORAGE, testName)
    np.savez(resName, **resultsDict)


def loadMultiRes(testName):
    resName = "%s/%s.npz" % (STORAGE, testName)
    if not os.path.exists(resName):
        print("Can't find result %s, giving up" % testName)
        sys.exit(1)

    results = np.load(resName)
    return {k: results[k] for k in results.keys()}


def get_tests(args):
    # Return name and tests to run
    if args.alltests:  # running all tests
        return "all tests", ["t%d" % i for i in range(1, 21)]
    elif args.allwarmups:  # running all warmups
        return "warmup tests", ["w%d" % i for i in range(1, 21)]
    elif args.allconvs:  # running all conv tests
        if not common.CONV_TESTS:
            print("If you want the convolution tests, "
                  "set CONV_TESTS in common.py to true")
            sys.exit(1)
        return "convolution tests", ["c%d" % i for i in range(1, 5)]
    return "some tests", args.test.split(",")


def get_test_fn(test_name):
    gen_fn = getattr(common, 'gen_' + test_name, None)
    if test_name[0] == 't':
        test_fn = getattr(tests, test_name, None)
    elif test_name[0] == 'w':
        test_fn = getattr(warmups, test_name, None)
    if gen_fn is None or test_fn is None:
        raise ValueError('Function "%s" does not exist!' % test_name)
    return test_fn, gen_fn


def main():
    args = parse_args()
    testName, toRun = get_tests(args)

    successes = 0
    for fnName in toRun:
        print("Running %s" % fnName)

        test_fn, gen_fn = get_test_fn(fnName)

        success = True
        resultsDict = {}
        for seed in SEEDS:
            np.random.seed(seed)
            data = gen_fn()

            try:
                # If gen_fn returns a tuple, spread them as positional args
                if isinstance(data, tuple):
                    res = test_fn(*data)
                else:
                    res = test_fn(data)
            except Exception as exc:
                print("Crashed! On %s seed %d" % (fnName, seed))
                print(traceback.format_exc())
                print(exc)
                success = False
                break

            if res is None:
                print("Not implemented %s or returned None on seed %d"
                      % (fnName, seed))
                success = False
                break

            resultsDict[str(seed)] = res

        # Don't bother if it didn't work
        if not success:
            continue

        if args.store:
            storeMultiRes(fnName, resultsDict)

        solnDict = loadMultiRes(fnName)
        success = True

        for seed in SEEDS:
            strSeed = str(seed)
            res, soln = resultsDict[strSeed], solnDict[strSeed]

            if res.shape != soln.shape:
                print("\tWrong shape! On %s seed %d" % (fnName, seed))
                print("\tGiven: %s" % str(res.shape))
                print("\tExpected: %s" % str(soln.shape))
                if args.pdb:
                    print("\tCredited response: soln\n\tYour response: res")
                    pdb.set_trace()
                success = False
                break

            if fnName == "t2":
                # Force signs to match for eigenvector problem
                res *= np.sign(res[0]) * np.sign(soln[0])

            same = np.allclose(res, soln, rtol=1e-3, atol=1e-4)
            if not same and fnName != 'w8':
                print("\tWrong values! On %s seed %d" % (fnName, seed))
                if args.pdb:
                    print("\tCredited response: soln\n\tYour response: res")
                    pdb.set_trace()
                success = False
                break

            if res.dtype.kind != soln.dtype.kind:
                print("\tNot the same kind! On %s seed %d" % (fnName, seed))
                print("\tGiven: %s" % res.dtype.name)
                print("\tExpected: %s" % soln.dtype.name)
                success = False
                break

        successes += 1 if success else 0

    frac = 100.0 * successes / len(toRun)
    print("Ran %s" % testName)
    print("%d/%d = %.1f" % (successes, len(toRun), frac))


if __name__ == '__main__':
    main()
