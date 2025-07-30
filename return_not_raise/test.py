"""
A demo script where we look at whether we can do some wacky stuff to approximate something
that looks like a result type.
"""

from typing import Callable


def return_not_raise(func: Callable) -> Callable:
    """
    A decorator that returns the result of the function instead of raising an exception.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e

    return wrapper


@return_not_raise
def stupid_function(x: int) -> int:
    """
    A function that raises an exception if x is less than 0.
    """
    if x < 0:
        raise ValueError("x must be non-negative")
    return x * 2


    main()
