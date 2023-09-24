import argparse
from env import TurbochargedDieselEngine
from NN import phi
import numpy as np
from scipy.integrate import solve_ivp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for PN-ADP')
    parser.add_argument('--xn', type=int, default=6, help='Dimension of the state')
    parser.add_argument('--un', type=int, default=2, help='Dimension of the input')
    parser.add_argument('--num_points', type=int, default=11,
                        help='Num of the points for integration')
    parser.add_argument('--N', type=int, default=10,
                        help='Num of the row of the LS matrix, should be at least xn^2+2*xn*un')
    parser.add_argument('--T', type=float, default=0.1,
                        help='Time inverval')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='end condition for policy iteration')
    parser.add_argument('--max_iteration', type=int, default=120,
                        help='max iteration number')
    parser.add_argument('--phin', type=int, default=36, help='Dimension of the phi')

    args = parser.parse_args()
    args_dict = vars(args)

    env = TurbochargedDieselEngine()

    # initialization
    N = args_dict['N']
    T = args_dict['T']
    num_points = args_dict['num_points']
    xn = args_dict['xn']
    phin = args_dict['phin']
    epsilon = args_dict['epsilon']
    max_iteration = args_dict['max_iteration']


    def policy(x):
        return np.atleast_1d(-K @ x)


    def sample_env(t, X):
        x = X[0:xn]
        u = policy(x)
        x_dot = env.A @ x + env.B @ u
        l_dot = np.atleast_1d(x.T @ env.Q @ x + u.T @ env.R @ u)
        return np.concatenate((x_dot, l_dot), axis=0)


    sample_interval = T / (num_points - 1)
    K = env.K0
    w = np.ones((phin, 1))
    w_old = np.Inf * np.ones((phin, 1))

    iteration = 0
    while np.linalg.norm(w - w_old) >= epsilon and iteration < max_iteration:
        print()
        print('iter = :', iteration)
        print('error = :', np.linalg.norm(w - w_old))
        # collect data
        I = np.zeros((N, 1))
        Phi = np.zeros((N, phin))

        X0 = np.append(env.x0, 0)
        for i in range(N):
            # init_t = 0
            init_t = i * T  # + iteration * N * sample_interval
            t_eval = np.arange(init_t, init_t + T + 1e-10, sample_interval)
            t_span = [t_eval[0], t_eval[-1]]
            sol = solve_ivp(sample_env, t_span=t_span, y0=X0, method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-8)
            X0 = sol.y[:, -1]  # y: (xn+1, length)

            x_seq = sol.y[0:-1, :]
            u_seq = np.apply_along_axis(policy, axis=0, arr=x_seq)

            l_seq = []

            for t in range(x_seq.shape[1]):
                x_t = x_seq[:, t]
                u_t = u_seq[:, t]
                l_seq.append(x_t.T @ env.Q @ x_t + u_t.T @ env.R @ u_t)

            l_seq = np.array(l_seq)

            integral = np.trapz(l_seq, t_eval)
            integral1 = sol.y[-1, -1] - sol.y[-1, 0]

            I[i, :] = integral1

            phi_end = phi(x_seq[:, -1])
            phi_start = phi(x_seq[:, 0])

            Phi[i, :] = phi_start - phi_end

        w_old = w
        w = np.linalg.pinv(Phi) @ I
        print('rank : ', np.linalg.matrix_rank(Phi))
        P = w.reshape(xn, xn)
        K = np.linalg.inv(env.R) @ env.B.T @ P
        print('K = ', K)
        print('P = ', P)

        iteration += 1
