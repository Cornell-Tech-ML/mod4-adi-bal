"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Return x * y"""
    return x * y


def id(x: float) -> float:
    """Return x"""
    return x


def add(x: float, y: float) -> float:
    """Return x + y"""
    return x + y


def neg(x: float) -> float:
    """Return -x"""
    return -x


def lt(x: float, y: float) -> bool:
    """Return 1.0 if x < y else 0.0"""
    return x < y


def eq(x: float, y: float) -> float:
    """Return 1.0 if x == y else 0.0"""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Return True if |x - y| < 1e-2 else False"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of x"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Return the natural logarithm of x"""
    return math.log(x)


def exp(x: float) -> float:
    """Return e raised to the power of x"""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the derivative of log(x) with respect to x"""
    return y / x


def inv(x: float) -> float:
    """Return 1.0 / x"""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Return the derivative of 1/x with respect to x"""
    return (-1.0 / (x**2)) * y


def relu_back(x: float, y: float) -> float:
    """Return the derivative of ReLU(x) with respect to x"""
    return y if x > 0 else 0.0


def map(fn: Callable[[float], float], arr: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element in an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.
        arr: An iterable of floats.

    Returns:
    -------
        A list where each element is the result of applying fn to the corresponding element of arr.

    """
    return [fn(x) for x in arr]


def zipWith(
    fn: Callable[[float, float], float], arr1: Iterable[float], arr2: Iterable[float]
) -> Iterable[float]:
    """Applies a function to pairs of elements from two iterables.

    Args:
    ----
        fn: A binary function that takes two floats and returns a float.
        arr1: The first iterable of floats.
        arr2: The second iterable of floats.

    Returns:
    -------
        A list where each element is the result of applying fn to corresponding elements of arr1 and arr2.

    """
    return [fn(x, y) for x, y in zip(arr1, arr2)]


def reduce(
    fn: Callable[[float, float], float], arr: Iterable[float], init: float
) -> float:
    """Reduces an iterable to a single value by repeatedly applying a binary function.

    Args:
    ----
        fn: A binary function that takes two floats and returns a float.
        arr: An iterable of floats to reduce.
        init: The initial value to start the reduction.

    Returns:
    -------
        A single float value that is the result of reducing arr under the binary function fn, starting from init.

    """
    acc = init
    for x in arr:
        acc = fn(acc, x)
    return acc


def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negates all elements in an iterable.

    Args:
    ----
        arr: An iterable of floats.

    Returns:
    -------
        A list of floats where each element is the negation of the corresponding element in arr.

    """
    return map(neg, arr)


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Adds elements from two iterables elementwise.

    Args:
    ----
        arr1: The first iterable of floats.
        arr2: The second iterable of floats.

    Returns:
    -------
        A list where each element is the sum of the corresponding elements of arr1 and arr2.

    """
    return zipWith(add, arr1, arr2)


def sum(arr: Iterable[float]) -> float:
    """Sums all elements in an iterable.

    Args:
    ----
        arr: An iterable of floats.

    Returns:
    -------
        A float that is the sum of all elements in arr.

    """
    return reduce(add, arr, 0.0)


def prod(arr: Iterable[float]) -> float:
    """Computes the product of all elements in an iterable.

    Args:
    ----
        arr: An iterable of floats.

    Returns:
    -------
        A float that is the product of all elements in arr.

    """
    return reduce(mul, arr, 1.0)
