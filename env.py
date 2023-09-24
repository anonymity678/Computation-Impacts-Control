import numpy as np

class ThirdOrderSystem:
    def __init__(self):
        a1 = 0.1
        a2 = 0.5
        a3 = 0.7
        self.A = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [-a1, -a2, -a3]])
        self.B = np.array([[0],
                           [0],
                           [1]])
        self.Q = np.eye(3)
        self.R = np.identity(1)
        self.K0 = np.zeros((1, 3))

        self.x0 = np.array([2, -2, 3])
        self.x0_env = np.array([2, -2, 3])
        self.Popt = np.array([[2.355030933091255, 2.238452370854351, 0.904987562112089],
                              [2.238452370854351, 4.241942496771635, 1.893095222031228],
                              [0.904987562112089, 1.893095222031228, 1.596995960828503]])

        self.Kopt = np.array([[0.904987562112088, 1.893095222031230, 1.596995960828503]])


class SinSystem:
    def __init__(self):
        self.x0 = np.array([1, 1])
        # self.x0_env = np.array([1, 1])
        self.x0_env = np.array([-1, 1])
        self.x0_env1 = np.array([-100, 100])

        self.w0 = np.array([[-1],
                            [3],
                            [1.5]])
        self.wopt = np.array([[0.5],
                              [0],
                              [1]])

        self.Q = np.identity(2)
        self.R = np.identity(1)

    @staticmethod
    def f(x):
        f1 = -x[0] + x[1]
        f2 = -1 / 2 * (x[0] + x[1]) + 1 / 2 * x[1] * np.sin(x[0]) ** 2
        return np.array([[f1], [f2]])

    @staticmethod
    def g(x):
        g1 = 0
        g2 = np.sin(x[0])
        return np.array([[g1], [g2]])

    def transition(self, x, u):
        u = np.atleast_2d(u)
        x_dot = self.f(x) + self.g(x) @ u
        return x_dot.flatten()

    @staticmethod
    def init_policy(x):
        u = -3 / 2 * np.sin(x[0]) * (x[0] + x[1])
        return u

    @staticmethod
    def d_phi(x):
        return np.array([[2 * x[0], x[1], 0], [0, x[0], 2 * x[1]]])