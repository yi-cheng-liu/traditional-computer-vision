import numpy as np

# Define some convenient aliases
gauss = lambda t: np.random.randn(*t)                          # noqa: E731
unif = lambda t: np.random.uniform(size=t)                     # noqa: E731
unifint = lambda l, h, t: np.random.randint(l, h + 1, size=t)  # noqa: E731

def frank(n=9, minSV=1, psd=True):
    # Return a full-rank nxn matrix with varying svs
    M = unif((n, n))
    M = M.T.dot(M) if psd else M
    U, S, VT = np.linalg.svd(M)
    for i in range(n):
        S[i] += (minSV + 1.0 / (i + 1))
    return U.dot(np.diag(S)).dot(VT)


def rot(n=3):
    M = unif((n, n))
    U, S, VT = np.linalg.svd(M)
    R = U.dot(VT)
    return np.linalg.det(R) * R


# Test-case generators
gen_w1 = lambda: gauss((10, 10))  # noqa: E731
gen_w2 = lambda: (gauss((10, 10)), gauss((10, 10)))  # noqa: E731
gen_w3 = lambda: (gauss((10, 10)), gauss((10, 10)))  # noqa: E731
gen_w4 = lambda: (gauss((10, 10)), gauss((10, 10)))  # noqa: E731
gen_w5 = lambda: np.round(unif((5, 5)) * 100)        # noqa: E731
gen_w6 = lambda: (unifint(1, 100, (10, 1)), np.ones((10, 1), dtype=np.int) * 100)  # noqa: E731, E501
gen_w7 = lambda: unif((10, 20))                                # noqa: E731
gen_w8 = lambda: np.random.choice(20) + 30                     # noqa: E731
gen_w9 = lambda: unif((10, 10))                                # noqa: E731
gen_w10 = lambda: np.random.choice(10) + 10                    # noqa: E731
gen_w11 = lambda: (gauss((10, 20)), gauss((20, 1)))            # noqa: E731
gen_w12h = lambda s: (frank(n=s), gauss((s, 1)))               # noqa: E731
gen_w12 = lambda: gen_w12h(np.random.choice(10) + 5)           # noqa: E731
gen_w13h = lambda s: (gauss((s, 1)), gauss((s, 1)))            # noqa: E731
gen_w13 = lambda: gen_w13h(np.random.choice(10) + 5)           # noqa: E731
gen_w14 = lambda: gauss((np.random.choice(10) + 5, 1))         # noqa: E731
gen_w15h = lambda s: (gauss((s, 3 * s)), np.random.choice(s))  # noqa: E731
gen_w15 = lambda: gen_w15h(np.random.choice(10) + 5)           # noqa: E731
gen_w16h = lambda s: gauss((s, s + 2))                         # noqa: E731
gen_w16 = lambda: gen_w16h(np.random.choice(4) + 2)            # noqa: E731
gen_w17 = gen_w16
gen_w18 = gen_w16
gen_w19 = gen_w16
gen_w20 = gen_w16

gen_t1 = lambda: [unif((1, 4)) for i in range(10)]  # noqa: E731
gen_t2 = frank
gen_t3 = lambda: unif((4, 3)) * 2 - 1       # noqa: E731
gen_t4 = lambda: (rot(), unif((10, 3)))     # noqa: E731
gen_t5 = lambda: unif((100, 100))           # noqa: E731
gen_t6 = lambda: np.random.choice(20) + 20  # noqa: E731
gen_t7 = lambda: gauss((10, 10))            # noqa: E731
gen_t8 = lambda: gauss((10, 10))            # noqa: E731
gen_t9 = lambda: (gauss((1, 10)), gauss((100, 10)), gauss((100, 1)))  # noqa: E731, E501
gen_t10 = lambda: [gauss((10, 4)) for i in range(30)]  # noqa: E731
gen_t11 = lambda: gauss((100, 4))                      # noqa: E731
gen_t12 = lambda: (gauss((100, 4)), gauss((200, 4)))   # noqa: E731
gen_t13 = lambda: (gauss((1, 4)), gauss((30, 4)))      # noqa: E731
gen_t14 = lambda: (gauss((100, 10)), gauss((100, 1)))  # noqa: E731
gen_t15 = lambda: (gauss((100, 3)), gauss((100, 3)))   # noqa: E731
gen_t16 = lambda: gauss((100, 3))                      # noqa: E731
gen_t17 = lambda: gauss((100, 4))                      # noqa: E731
gen_t18 = lambda: (100, 10, np.random.choice(80) + 10, np.random.choice(80) + 10)  # noqa: E731, E501
gen_t19 = lambda: (100, 10, np.random.choice(80) + 10, np.random.choice(80) + 10)  # noqa: E731, E501
gen_t20 = lambda: (100, np.array([1, -1, 0]))  # noqa: E731

gen_c1 = lambda: (gauss((10, 10)), 3)    # noqa: E731
gen_c2 = lambda: (gauss((10, 10)), 3)    # noqa: E731
gen_c3 = lambda: (np.ones((12, 12)), 2)  # noqa: E731
gen_c4 = lambda: (gauss((100, 100)), 7)  # noqa: E731
