import sys
from scipy.optimize import minimize
import numpy as np
import pandas as pd


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = X
        self.train_y = y

        # hyperparameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[
                1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


if __name__ == '__main__':
    # sys.argv
    # input: for train ./bayesian.py patternType style tilingType cutStyle Q Xmin Ymin scale Value
    #        for test  ./bayesian.py patternType style tilingType cutStyle Q Xmin Ymin scale
    # model = BayesianRegressor(alpha=2e-3, beta=2)

    # 获取数据， 如果是新增参数， 就写入txt文件， 再运行
    args = len(sys.argv)
    gpr = GPR(optimize=False)
    if args == 10:
        gpr.optimize = False
        data = pd.DataFrame(
            [[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7],
              sys.argv[8],sys.argv[9]]])
        data.to_csv('OptimizationRecord.txt', mode='a', sep=' ', header=False, index=False)

        data = pd.read_table('OptimizationRecord.txt', header=None)
        X = np.array(np.split(data))
        trainx = X[:, 0:7]
        trainy = X[:, 8:9].T
        gpr.fit(trainx, trainy)

        m = 0
        c = 0
        return_data = pd.DataFrame[[0, 0, 0, 0, 0, 0, 0, 0]]
        for pt in range(0, 1):
            for st in range(0, 111):
                for tt in range(0, 211):
                    for cs in (0, 42):
                        for q in np.arange(0.0, 10.0, 0.001):
                            for xmin in np.arange(-10, 10, 0.001):
                                for ymin in np.arange(-10, 10, 0.001):
                                    for scale in range(10, 20):
                                        data = pd.DataFrame([[pt, st, tt, cs, q, scale]])
                                        nm, nc = gpr.predict(data)
                                        if nc > c:
                                            c = nc
                                            return_data = data
        # TODO： 这里的参数需要根据需求修改
        if c < 0.1:
            print("The model has been trained")
        else:
            print(return_data.loc[0].to_string(index=False))

    elif args == 8:
        newDate = pd.DataFrame[[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]]
        m, c = gpr.predict(newDate)
        print(m)
        print(c)

    else:
        print("参数输入错误");
