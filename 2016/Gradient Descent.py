def problem(x):
    e = 2.71828182845904590
    return x ** 3 + 2 * x + e ** x - 3


def error(x):
    pointwewant = 1
    return (problem(x) - pointwewant) ** 2


def derivative_descent(x):
    vSNum = 0.0001 # Very Small Numer for the delta in the derivative equation
    derivative = (error(x + vSNum) - error(x - vSNum)) / (2 * vSNum)

    middleMan = 0.01  # This man can't be too big or too small because the decent will be either moving slow or overshoot.
    x = x - derivative * middleMan # I think it is better to write it as -derivative * middleMan+x
    return x


x = 0.0
for i in range(50):
    x = derivative_descent(x)
    print('x ={:6f}, problem(x) = {:6f}, Error = {:6f}'.format(x, problem(x), error(x)))
