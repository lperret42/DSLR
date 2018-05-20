from src.utils import is_int, quicksort

def abs(x):
    return -x if x < 0 else x

def min(lst):
    m = lst[0]
    for elem in lst:
        if elem < m:
            m = elem
    return m

def max(lst):
    m = lst[0]
    for elem in lst:
        if elem > m:
            m = elem
    return m

def sum(lst):
    s = 0
    for elem in lst:
        s += elem
    return s

def mean(lst):
    return sum(lst) / len(lst)

def std(lst):
    mu = mean(lst)
    return sqrt(sum([(x - mu) ** 2 for x in lst]) / (len(lst) - 1))

def ceil(f):
    return int(f) if is_int(f) or f < 0 else int(f) + 1

def quartile_n(lst, n):
    return lst[ceil(float(n * len(lst)) / 4.)]

def linear_function(a, b, x):
    return a * x + b

def sqrt(x, epsilon=10e-15):
    """
    implementation of the sqrt function, both suites u and v converge to sqrt(x)
    """
    if x < 0:
        return None
    if x == 0:
        return 0
    u = 1
    v = x
    error_u = abs(u * u - x)
    error_v = abs(v * v - x)
    old_error_u = error_u
    old_error_v = error_v
    while error_u > epsilon and error_v > epsilon:
        tmp = u
        u = 2. / (1. / u + 1. / v)
        v = (tmp + v) / 2.
        error_u = abs(u * u - x)
        error_v = abs(v * v - x)
        if old_error_u == error_u and old_error_v == old_error_v:
            break
        old_error_u = error_u
        old_error_v = error_v

    return u if error_u <= error_v else v
