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


INT_METHOD = {'Euler': _euler, 'RK4': _runge_kutta, 'Pred-corr': _pred_correct,
              'Central diff': _central_diff}


def integrate(method, h, deriv, *y_vals):
    func = INT_METHOD[method]
    return func(h, deriv, *y_vals)