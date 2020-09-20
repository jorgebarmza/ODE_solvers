import solvers


def solve_with_all(p_num, problem_args_dict, method_args_dict):
    for name in ['euler', 'heun', 'rk4']:
        full_dict = {}
        full_dict.update(problem_args_dict)
        full_dict.update(method_args_dict[name])
        full_dict['filename'] = 'results/p{}_{}.txt'.format(p_num, name)

        solvers.approximate(**full_dict)


if __name__ == "__main__":

    # Precision requirements.
    euler_args = dict(
        method=solvers.euler,
        decimals=3,
        n_init=50,
    )

    heun_args = dict(
        method=solvers.heun,
        decimals=5,
        n_init=10,
    )

    rk4_args = dict(
        method=solvers.rk4,
        decimals=9,
        n_init=10,
    )

    method_args_dict = {
        'euler': euler_args,
        'heun': heun_args,
        'rk4': rk4_args
    }

    # Problem 1.
    e = 2.71828182846
    p1_args = dict(
        target=e,
        f=lambda x, y: y,
        x_init=0,
        y_init=1,
        x_final=1,
    )

    # Problem 2.
    ln2 = 0.69314718056
    p2_args = dict(
        target=ln2,
        f=lambda x, y: 1 / x,
        x_init=1,
        y_init=0,
        x_final=2,
    )

    # Problem 3.
    pi = 3.14159265359
    p3_args = dict(
        target=pi,
        f=lambda x, y: 4 / (1 + x**2),
        x_init=0,
        y_init=0,
        x_final=1,
    )

    # Solve all problems with all methods.
    for p_num, p_args in zip([1, 2, 3], [p1_args, p2_args, p3_args]):
        solve_with_all(p_num, p_args, method_args_dict)
