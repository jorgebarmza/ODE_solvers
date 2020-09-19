import solvers

if __name__ == "__main__":

    # Problem 1.
    e = 2.71828182846
    p1_args = dict(
        target=e,
        f=lambda x, y: y,
        x_init=0,
        y_init=1,
        x_final=1,
    )

    p1_euler_args = p1_args
    p1_euler_args.update(
        dict(
            filename='p1_euler.txt',
            method=solvers.euler,
            decimals=3,
            n_init=50,
        ))

    p1_heun_args = p1_args
    p1_heun_args.update(
        dict(
            filename='zp1_heun.txt',
            method=solvers.heun,
            decimals=5,
            n_init=10,
        ))

    p1_rk4_args = p1_args
    p1_rk4_args.update(
        dict(
            filename='p1_rk4.txt',
            method=solvers.rk4,
            decimals=9,
            n_init=10,
        ))

    solvers.approximate(**p1_rk4_args)
