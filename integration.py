#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# integration.py:
# Here we define the different numerical integration methods, such as
# Euler, Runge-kutta, predictor-corrector and central difference.
# These methods will be used in different files.


def _euler(h, deriv, y):
    k = deriv(y)
    next_y = y + h * k
    return next_y, k


def _runge_kutta(h, deriv, y):
    k1 = deriv(y)
    k2 = deriv(y + h * k1/2)
    k3 = deriv(y + h * k2/2)
    k4 = deriv(y + h * k3)
    next_y = y + (k1 + 2*k2 + 2*k3 + k4) * h/6
    return next_y, k1


def _pred_correct(h, deriv, y):
    k1 = deriv(y)
    next_y = y + h * k1
    k2 = deriv(next_y)
    second_y = y + h/2 * (k1 + k2)
    return second_y, k1


def _central_diff(h, deriv, y, prev_y):
    k = deriv(y)
    next_y = prev_y + 2 * h * k
    return next_y, k


INT_METHOD = {'euler': _euler, 'rk4': _runge_kutta, 'pred-corr': _pred_correct,
              'central diff': _central_diff}


def integrate(method, h, deriv, *y_vals):
    func = INT_METHOD[method]
    return func(h, deriv, *y_vals)
