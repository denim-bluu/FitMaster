import numpy as np
from fitmaster.forms.concrete import LinearForm, ExponentialForm, LogarithmicForm
from numpy.testing import assert_almost_equal


def test_linear_form():
    linear_form = LinearForm()
    x = np.array([1.0, 2.0, 3.0])
    a = 1.0
    b = 2.0
    assert_almost_equal(linear_form.func(x, a, b), np.array([3.0, 5.0, 7.0]))
    assert linear_form.initial_guess(x, x) == [1, 1]


def test_exponential_form():
    exponential_form = ExponentialForm()
    x = np.array([1.0, 2.0, 3.0])
    a = 1.0
    b = 2.0
    c = 1.0
    assert_almost_equal(
        exponential_form.func(x, a, b, c),
        np.array([8.3890561, 55.59815003, 404.42879349]),
    )
    assert exponential_form.initial_guess(x, x) == [1, 1, 1]


def test_logarithmic_form():
    logarithmic_form = LogarithmicForm()
    x = np.array([1.0, 2.0, 3.0])
    a = 1.0
    b = 2.0
    assert_almost_equal(
        logarithmic_form.func(x, a, b),
        np.array([1.0, 2.38629436, 3.19722458]),
    )
    assert logarithmic_form.initial_guess(x, x) == [1, 1]
