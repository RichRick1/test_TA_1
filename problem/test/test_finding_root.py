from ..finding_root import *

import numpy as np
from numpy.testing import assert_equal, assert_allclose

def test_f():
    """
    testing f(x) function
    :return:
    """
#     print(f(-1))
    assert_allclose(1, f(1))
    assert_allclose(2, f(8))
#     assert_allclose(-1, f(-1))


def test_exponential_function():
    """
    testing exponential function
    :return:
    """
    assert_allclose(-0.5, exponential_function(0))
