#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# integration.py:
# Here we define the different numerical integration methods, such as
# Euler, Runge-kutta, predictor-corrector and central difference.
# These methods will be used in different files.


def _euler(h, deriv, y):
    """Numerical method called Euler which is calculated by
    y_{n + 1} = y_n + h * f(t_n, y_n). Where f is the function _diff from
    diver.py.
    """
    k = deriv(y)
    next_y = y + h * k
    return next_y, k


def _runge_kutta(h, deriv, y):
    """Numerical method called Runge-Kutta order 4, which is calculated by
    y_{n + 1} = y_n + 1/6 * (k_1 + 2k_2 + 2k_3 + k_4) * h
    Where k_1 = f(t_n, y_n), k_2 = f(t_n + h/2, y_n + h * k1/2),
    k3 = f(t_n + h/2, y_n + h * k2/2) and k4 = f(t_n + h, y_n + h * k2).
    """
    k1 = deriv(y)
    k2 = deriv(y + h * k1/2)
    k3 = deriv(y + h * k2/2)
    k4 = deriv(y + h * k3)
    next_y = y + (k1 + 2*k2 + 2*k3 + k4) * h/6
    return next_y, k1


def _pred_correct(h, deriv, y):
    """Numerical method called Predictor-corrector which is calculated by
    first euler and then using the trapezoidal rule. The method is given by
    y_{n + 1} = y_n + 1/2 * h * (f(t_n, y_n) + f(t_{n + 1}, k_{n + 1})) were
    k_{n + 1} is Euler.
    """
    k1 = deriv(y)
    next_y = y + h * k1
    k2 = deriv(next_y)
    second_y = y + h/2 * (k1 + k2)
    return second_y, k1


def _central_diff(h, deriv, y, prev_y):
    """Numerical method called central difference."""
    k = deriv(y)
    next_y = prev_y + 2 * h * k
    return next_y, k


# Give the methods easier names to work with.
INT_METHOD = {'euler': _euler, 'rk4': _runge_kutta, 'pred-corr': _pred_correct,
              'central diff': _central_diff}


def integrate(method, h, deriv, *y_vals):
    """Function is used to call the different methods defined above."""
    func = INT_METHOD[method]
    return func(h, deriv, *y_vals)
