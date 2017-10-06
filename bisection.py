import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def bisection(f, lower_bound, upper_bound, TABLE=False, TOL=0.001, NMAX=100):

    if f(lower_bound) * f(upper_bound) > 0:
        print("No Sign Change Try New Boundries")
        quit()

    iter = 0

    error_eval = 1

    TABLE = np.zeros((NMAX, 5))

    estimate = (lower_bound + upper_bound) / 2

    TABLE[:, 0] = np.arange(1, NMAX + 1)

    TABLE[0, 1::] = [lower_bound, upper_bound, estimate, error_eval]

    while iter <= NMAX:

        if f(estimate) == 0 or error_eval < TOL:

            TABLE = TABLE[:iter + 1, :]
            solution = TABLE[iter, 3]

            if TABLE == True:

                df = pd.DataFrame(data=TABLE[::, 1:])
                df.columns = ['Left', 'Right', 'Estimate', 'Error']
                df.index = np.arange(1, TABLE.shape[0] + 1)
                pd.set_option("display.colheader_justify", "left")

                return df

            if TABLE == False:
                return solution

        else:
            iter = iter + 1

            if f(lower_bound) * f(estimate) < 0:
                upper_bound = estimate

            elif f(lower_bound) * f(estimate) > 0:
                lower_bound = estimate

        new_estimate = (lower_bound + upper_bound) / 2

        error_eval = abs((new_estimate - estimate) / new_estimate)

        estimate = new_estimate

        TABLE[iter, 1:] = [lower_bound, upper_bound, estimate, error_eval]

    return False


# def func(x):
#     return math.pow(x, 3) - x - 2
#
#
# func_vec = np.vectorize(func, otypes=[np.float])
#
# sol = bisection(func, -10, 10, TAB=1)
#
#
# sol = sol.as_matrix()
#
# sol = np.column_stack((np.arange(1,sol.shape[0]+1),sol))
#
#
# print(sol)
#
# for picnum in np.arange(0, sol.shape[0]):
#
#     Left = sol[picnum, 1]
#     Right = sol[picnum, 2]
#     root = sol[picnum, 3]
#     span = abs(Right - root) + abs(Left - root)
#     padding = 0.055 * span
#
#     x_data = np.arange(Left - padding, Right + padding, 0.0001)
#
#     y_data = func_vec(x_data)
#
#     plot = plt.plot(x_data, y_data)
#
#     plt.setp(plot, linewidth=0.5, color='g')
#
#     plt.axhline(linewidth=0.5, color="g")
#     # plt.axvline(linewidth=0.3, color="r")
#
#     plt.plot(Left, func(Left), marker='o', markersize=5, color="red")
#     plt.plot(Right, func(Right), marker='o', markersize=5, color="red")
#
#     plt.plot(Left, 0, marker='o', markersize=5, color="red")
#     plt.plot(Right, 0, marker='o', markersize=5, color="red")
#
#     plt.plot([Left, Left], [0, func(Left)])
#     plt.plot([Right, Right], [0, func(Right)])
#
#     plt.plot(root, 0, marker='o', markersize=5, color="red")
#
#     plt.show()