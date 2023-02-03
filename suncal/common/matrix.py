''' Matrix operations using Python lists so the operations may be done on Sympy
    expressions, not just numbers.
'''

# Note: sympy has Matrix object which would handle some of this, but
# it can't subs() Pint quantities, so the eval functions are still needed.

import sympy


def matmul(a, b):
    ''' Matrix multiply. Manually looped to preserve units since Pint
        doesn't allow matrices with different units on each element.

        Args:
            a: List of list of sympy expressions
            b: List of list of sympy expressions
    '''
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            product = 0
            for v in range(len(a[i])):
                if a[i][v] == 1:   # Symbolic gets ugly when multiplying by 1
                    product += b[v][j]
                elif b[v][j] == 1:
                    product += a[i][v]
                else:
                    product += a[i][v] * b[v][j]
            row.append(product)
        result.append(row)
    return result


def diagonal(a):
    ''' Return diagonal of square matrix

        Args:
            a: list of list of sympy expressions
    '''
    if len(a) > 0 and a[0]:
        return [a[i][i] for i in range(len(a))]
    else:
        return []


def transpose(a):
    ''' Transpose matrix (to preserve units)

        Args:
            a: list of list of sympy expressions
    '''
    return list(map(list, zip(*a)))


def eval_matrix(U, values):
    ''' Evaluate matrix (list of lists) of sympy expressions U with values

        Args:
            U: list of list of sympy expressions
            values: dictionary of {name:value} to substitute

        Returns:
            list of list of floats
    '''
    U_eval = []
    for row in U:
        U_row = []
        for expr in row:
            df = sympy.lambdify(values.keys(), expr, 'numpy')  # Can't subs() with pint Quantities
            U_row.append(df(**values))
        U_eval.append(U_row)
    return U_eval


def eval_list(U, values):
    ''' Evaluate a list of sympy expressions U with values

        Args:
            U: list of sympy expressions
            values: dictionary of {name:value} to substitute

        Returns:
            list of floats
    '''
    U_eval = []
    for expr in U:
        df = sympy.lambdify(values.keys(), expr, 'numpy')  # Can't subs() with pint Quantities
        U_eval.append(df(**values))
    return U_eval


def eval_dict(U, values):
    ''' Evaluate a dictionary of sympy expressions U with values

        Args:
            U: dictionary of {name:sympy expressions}
            values: dictionary of {name:value} to substitute

        Returns:
            dictionary of {name:float}
    '''
    U_eval = {}
    for name, expr in U.items():
        df = sympy.lambdify(values.keys(), expr, 'numpy')  # Can't subs() with pint Quantities
        U_eval[name] = df(**values)
    return U_eval
