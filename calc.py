# -*- coding: utf-8 -*-
# @Time : 2020/5/17 21:16
# @Author : mora
# @FileName: calc.py
# @Github ：https://github.com/intellectualmora/linjh
import numpy as np
import pandas as pd
from sko.GA import GA
import sys
import matplotlib.pyplot as plt

maxborder = 30
prec = 1e-6
NaN = sys.maxsize
path = "./data.csv"
zeroP = (147, 126)
maxiter = 800
popsize = 50

def importData():
    """
    :return: arr
    """
    df = pd.read_csv(path)
    t1 = np.array(df)
    return t1


def dataHandling(data):
    """
    :param data:
    :return:
    """
    dataRg = data.max(axis=0)
    r = np.zeros([int(dataRg[0] / 2) + 1, int(dataRg[1] / 2 + 1), 6])
    r[:, :, 0:3] = NaN
    r[:,:,5] = NaN
    for i in range(len(data)):
        r[int(data[i, 0] / 2), int(data[i, 1] / 2), 3] = data[i, 0]
        r[int(data[i, 0] / 2), int(data[i, 1] / 2), 4] = data[i, 1]
        r[int(data[i, 0] / 2), int(data[i, 1] / 2), 2] = data[i, 2]
        if data[i, 2] == 0:
            r[int(data[i, 0] / 2), int(data[i, 1] / 2), 0:2] = 0
            r[int(data[i, 0] / 2), int(data[i, 1] / 2), 5] = 0
    return r

def reconstruct(r):
    for border in range(1, maxborder):
        print('border is', border)
        searching(r, zeroP, border, zeroP[0] + border, zeroP[1] - border, 0)
    return

def searching(r, zeroP, border, kx, ky, sec):
    def _calR():
        def obj_func(p):
            x1, x2 = p
            return np.abs(r[kx, ky, 0] + x1 - x2 - r[kx, ky + 1, 1])
        if r[kx + 1, ky, 0] == NaN:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 0] - x[0]) - 0.1
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter,
                    lb=[-r[kx + 1, ky, 2], 0],
                    ub=[r[kx + 1, ky, 2] + prec, r[kx + 1, ky + 1, 2] + prec], precision=[prec, prec],
                    constraint_ueq=constraint_ueq)
        else:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 1] - x[0]) - 0.1,
                lambda x: np.abs(r[kx, ky+1, 0] - x[1]) - 0.1,
            ]
            constraint_eq = [
                lambda x: r[kx + 1, ky, 2] ** 2 - r[kx + 1, ky, 0] ** 2 - x[0] ** 2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[-r[kx + 1, ky, 2], 0],
                    ub=[r[kx + 1, ky, 2] + prec, r[kx + 1, ky + 1, 2] + prec], precision=[prec, prec],
                    constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        best_x, _ = ga.run()
        r[kx + 1, ky, 1] = best_x[0]
        r[kx + 1, ky + 1, 0] = best_x[1]
        return

    def _calT():
        def obj_func(p):
            x1, x2 = p
            return np.abs(r[kx, ky, 1] - x1 - x2 + r[kx - 1, ky, 0])

        if r[kx, ky + 1, 1] == NaN:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 1] - x[0]) - 0.2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[0, - r[kx - 1, ky + 1, 2]],
                    ub=[r[kx, ky + 1, 2]+ prec, r[kx - 1, ky + 1, 2]+ prec], precision=[prec, prec]
                    ,constraint_ueq=constraint_ueq)
        else:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 0] - x[0]) - 0.1,
                lambda x: np.abs(r[kx -1, ky, 1] - x[1]) - 0.1,
            ]
            constraint_eq = [
                lambda x: r[kx, ky + 1, 2] ** 2 - r[kx, ky + 1, 1] ** 2 - x[0] ** 2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[0, - r[kx - 1, ky + 1, 2]],
                    ub=[r[kx, ky + 1, 2]+ prec, r[kx - 1, ky + 1, 2]+ prec], precision=[prec, prec],
                    constraint_eq=constraint_eq, constraint_ueq = constraint_ueq)
        best_x, _ = ga.run()
        r[kx, ky + 1, 0] = best_x[0]
        r[kx - 1, ky + 1, 1] = best_x[1]
        return

    def _calL():
        def obj_func(p):
            x1, x2 = p
            return np.abs(-r[kx, ky, 0] - x1 + x2 + r[kx, ky - 1, 1])

        if r[kx - 1, ky, 0] == NaN:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 0] - x[0]) - 0.2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[-r[kx - 1, ky, 2], 0],
                    ub=[r[kx - 1, ky, 2]+ prec, r[kx - 1, ky - 1, 2]+ prec], precision=[prec, prec],constraint_ueq=constraint_ueq)
        else:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 1] - x[0]) - 0.1,
                lambda x: np.abs(r[kx, ky - 1, 0] - x[1]) - 0.1,
            ]
            constraint_eq = [
                lambda x: r[kx - 1, ky, 2] ** 2 - r[kx - 1, ky, 0] ** 2 - x[0] ** 2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[-r[kx - 1, ky, 2], 0],
                    ub=[r[kx - 1, ky, 2]+ prec, r[kx - 1, ky - 1, 2]+ prec], precision=[prec, prec],
                    constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        best_x, _ = ga.run()
        r[kx - 1, ky, 1] = best_x[0]
        r[kx - 1, ky - 1, 0] = best_x[1]
        return

    def _calD():
        def obj_func(p):
            x1, x2 = p
            return np.abs(-r[kx, ky, 1] + x1 + x2 - r[kx + 1, ky, 0])

        if r[kx, ky - 1, 1] == NaN:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 1] - x[0]) - 0.2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[0, -r[kx + 1, ky - 1, 2]],
                    ub=[r[kx, ky - 1, 2]+ prec, r[kx + 1, ky - 1, 2]+ prec], precision=[prec, prec],constraint_ueq=constraint_ueq)
        else:
            constraint_ueq = [
                lambda x: np.abs(r[kx, ky, 0] - x[0]) - 0.1,
                lambda x: np.abs(r[kx + 1, ky, 1] - x[1]) - 0.1
            ]
            constraint_eq = [
                lambda x: r[kx, ky - 1, 2] ** 2 - r[kx, ky - 1, 1] ** 2 - x[0] ** 2
            ]
            ga = GA(func=obj_func, n_dim=2, size_pop=popsize, max_iter=maxiter, lb=[0, -r[kx + 1, ky - 1, 2]],
                    ub=[r[kx, ky - 1, 2]+ prec, r[kx + 1, ky - 1, 2]+ prec], precision=[prec, prec],
                    constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        best_x, _ = ga.run()
        r[kx, ky - 1, 0] = best_x[0]
        r[kx + 1, ky - 1, 1] = best_x[1]
        return

    def _calE():
        # ==============
        def obj_func1(p):
            x1, = p

            assert r[zeroP[0] + border, zeroP[1] - border, 0] != NaN, "obj_func1 is NaN"
            assert r[zeroP[0] + border + 1, zeroP[1] - border, 1] != NaN, "obj_func1 is NaN"
            assert r[zeroP[0] + border, zeroP[1] - border - 1, 1] != NaN, "obj_func1 is NaN"

            return np.abs(
                r[zeroP[0] + border, zeroP[1] - border, 0] - r[zeroP[0] + border + 1, zeroP[1] - border, 1] - x1 + r[
                    zeroP[0] + border, zeroP[1] - border - 1, 1]
            )

        def obj_func2(p):
            x1, x2, x3 = p

            assert r[zeroP[0] + border, zeroP[1] - border, 1] != NaN, "obj_func2 is NaN"

            return np.abs(-r[zeroP[0] + border, zeroP[1] - border, 1] + x1 + x2 - x3)

        constraint_eq1 = [
            lambda x: r[zeroP[0] + border, zeroP[1] - border - 1, 2] ** 2 - r[
                zeroP[0] + border, zeroP[1] - border - 1, 1] ** 2 - x[0] ** 2,
            lambda x: r[zeroP[0] + border + 1, zeroP[1] - border - 1, 2] ** 2 - r[
                zeroP[0] + border + 1, zeroP[1] - border - 1, 0] ** 2 - x[1] ** 2,
            lambda x: r[zeroP[0] + border + 1, zeroP[1] - border, 2] ** 2 - r[
                zeroP[0] + border + 1, zeroP[1] - border, 1] ** 2 - x[2] ** 2
        ]
        ga = GA(func=obj_func1, n_dim=1, size_pop=popsize, max_iter=maxiter,
                lb=0,
                ub=r[zeroP[0] + border + 1, zeroP[1] - border - 1, 2] + prec,
                precision=prec)
        best_x, _ = ga.run()
        r[zeroP[0] + border + 1, zeroP[1] - border - 1, 0] = best_x[0]
        ga = GA(func=obj_func2, n_dim=3, size_pop=popsize, max_iter=maxiter,
                lb=[0,
                    -r[zeroP[0] + border + 1, zeroP[1] - border - 1, 2],
                    0],
                ub=[r[zeroP[0] + border, zeroP[1] - border - 1, 2]+ prec,
                    r[zeroP[0] + border + 1, zeroP[1] - border - 1, 2]+ prec,
                    r[zeroP[0] + border + 1, zeroP[1] - border, 2]+ prec],
                precision=[prec, prec, prec], constraint_eq=constraint_eq1)
        best_x, _ = ga.run()
        r[zeroP[0] + border, zeroP[1] - border - 1, 0] = best_x[0]
        r[zeroP[0] + border + 1, zeroP[1] - border - 1, 1] = best_x[1]
        r[zeroP[0] + border + 1, zeroP[1] - border, 0] = best_x[2]

        # ==============
        def obj_func3(p):
            x1, = p

            assert r[zeroP[0] + border, zeroP[1] + border, 1] != NaN, "obj_func3 is NaN"
            assert r[zeroP[0] + border, zeroP[1] + border + 1, 0] != NaN, "obj_func3 is NaN"
            assert r[zeroP[0] + border + 1, zeroP[1] + border, 0] != NaN, "obj_func3 is NaN"

            return np.abs(
                r[zeroP[0] + border, zeroP[1] + border, 1] + r[zeroP[0] + border, zeroP[1] + border + 1, 0]
                - x1 - r[zeroP[0] + border + 1, zeroP[1] + border, 0]
            )

        def obj_func4(p):
            x1, x2, x3 = p

            assert r[zeroP[0] + border            , zeroP[1] + border, 0] != NaN, "obj_func4 is NaN"

            return np.abs(
                r[zeroP[0] + border, zeroP[1] + border, 0] + x1 - x2 - x3
            )

        constraint_eq2 = [
            lambda x: r[zeroP[0] + border + 1, zeroP[1] + border, 2] ** 2 - r[
                zeroP[0] + border + 1, zeroP[1] + border, 0] ** 2 - x[0] ** 2,
            lambda x: r[zeroP[0] + border + 1, zeroP[1] + border + 1, 2] ** 2 - r[
                zeroP[0] + border + 1, zeroP[1] + border + 1, 1] ** 2 - x[1] ** 2,
            lambda x: r[zeroP[0] + border, zeroP[1] + border + 1, 2] ** 2 - r[
                zeroP[0] + border, zeroP[1] + border + 1, 0] ** 2 - x[2] ** 2
        ]

        ga = GA(func=obj_func3, n_dim=1, size_pop=popsize, max_iter=maxiter,
                lb=-r[zeroP[0] + border + 1, zeroP[1] + border + 1, 2],
                ub=r[zeroP[0] + border + 1, zeroP[1] + border + 1, 2]+ prec, precision=prec)
        best_x, _ = ga.run()
        r[zeroP[0] + border + 1, zeroP[1] + border + 1, 1] = best_x[0]
        ga = GA(func=obj_func4, n_dim=3, size_pop=popsize, max_iter=maxiter,
                lb=[-r[zeroP[0] + border + 1, zeroP[1] + border, 2],
                    0,
                    -r[zeroP[0] + border, zeroP[1] + border + 1, 2]],
                ub=[r[zeroP[0] + border + 1, zeroP[1] + border, 2]+ prec,
                    r[zeroP[0] + border + 1, zeroP[1] + border + 1, 2]+ prec,
                    r[zeroP[0] + border, zeroP[1] + border + 1, 2]+ prec],
                precision=[prec, prec, prec], constraint_eq=constraint_eq2)
        best_x, _ = ga.run()
        r[zeroP[0] + border + 1, zeroP[1] + border, 1] = best_x[0]
        r[zeroP[0] + border + 1, zeroP[1] + border + 1, 0] = best_x[1]
        r[zeroP[0] + border, zeroP[1] + border + 1, 1] = best_x[2]

        # ==============
        def obj_func5(p):
            x1, = p

            assert r[zeroP[0] - border, zeroP[1] + border, 0] != NaN, "obj_func5 is NaN"
            assert r[zeroP[0] - border - 1, zeroP[1] + border, 1] != NaN, "obj_func5 is NaN"
            assert r[zeroP[0] - border, zeroP[1] + border + 1, 1] != NaN, "obj_func5 is NaN"

            return np.abs(
                -r[zeroP[0] - border, zeroP[1] + border, 0] + r[zeroP[0] - border - 1, zeroP[1] + border, 1] + x1 - r[
                    zeroP[0] - border, zeroP[1] + border + 1, 1]
            )

        def obj_func6(p):
            x1, x2, x3 = p

            assert r[zeroP[0] - border, zeroP[1] + border, 1] != NaN, "obj_func6 is NaN"

            return np.abs(
                r[zeroP[0] - border, zeroP[1] + border, 1] - x1 - x2 + x3
            )

        constraint_eq3 = [
            lambda x: r[zeroP[0] - border, zeroP[1] + border + 1, 2] ** 2 - r[
                zeroP[0] - border, zeroP[1] + border + 1, 1] ** 2 - x[0] ** 2,
            lambda x: r[zeroP[0] - border - 1, zeroP[1] + border + 1, 2] ** 2 - r[
                zeroP[0] - border - 1, zeroP[1] + border + 1, 0] ** 2 - x[1] ** 2,
            lambda x: r[zeroP[0] - border - 1, zeroP[1] + border, 2] ** 2 - r[
                zeroP[0] - border - 1, zeroP[1] + border, 1] ** 2 - x[2] ** 2
        ]

        ga = GA(func=obj_func5, n_dim=1, size_pop=popsize, max_iter=maxiter,
                lb=0,
                ub=r[zeroP[0] + border - 1, zeroP[1] + border + 1, 2]+ prec, precision=prec)
        best_x, _ = ga.run()
        r[zeroP[0] - border - 1, zeroP[1] + border + 1, 0] = best_x[0]
        ga = GA(func=obj_func6, n_dim=3, size_pop=popsize, max_iter=maxiter,
                lb=[0,
                    -r[zeroP[0] - border - 1, zeroP[1] + border + 1, 2],
                    0],
                ub=[r[zeroP[0] - border, zeroP[1] + border + 1, 2]+ prec,
                    r[zeroP[0] - border - 1, zeroP[1] + border + 1, 2]+ prec,
                    r[zeroP[0] - border - 1, zeroP[1] + border, 2]+ prec],
                precision=[prec, prec, prec], constraint_eq=constraint_eq3)
        best_x, _ = ga.run()
        r[zeroP[0] - border, zeroP[1] + border + 1, 0] = best_x[0]
        r[zeroP[0] - border - 1, zeroP[1] + border + 1, 1] = best_x[1]
        r[zeroP[0] - border - 1, zeroP[1] + border, 0] = best_x[2]

        # ==============

        def obj_func7(p):
            x1, = p

            assert r[zeroP[0] - border, zeroP[1] - border, 1] != NaN, "obj_func7 1 is NaN"
            assert r[zeroP[0] - border, zeroP[1] - border - 1, 0] != NaN, "obj_func7 2 is NaN"
            assert r[zeroP[0] - border - 1, zeroP[1] - border, 0] != NaN, "obj_func7 3 is NaN"


            return np.abs(
                -r[zeroP[0] - border, zeroP[1] - border, 1] - r[zeroP[0] - border, zeroP[1] - border - 1, 0] + x1 + r[
                    zeroP[0] - border - 1, zeroP[1] - border, 0]
            )

        def obj_func8(p):
            x1, x2, x3 = p

            assert r[zeroP[0] - border, zeroP[1] - border, 0] != NaN, "obj_func8 is NaN"

            return np.abs(
                -r[zeroP[0] - border, zeroP[1] - border, 0] - x1 + x2 + x3
            )

        constraint_eq4 = [
            lambda x: r[zeroP[0] - border -1, zeroP[1] - border, 2] ** 2
                      -r[zeroP[0] - border -1, zeroP[1] - border, 0] ** 2 - x[0] ** 2,
            lambda x: r[zeroP[0] - border - 1, zeroP[1] - border - 1, 2] ** 2 - r[
                zeroP[0] - border - 1, zeroP[1] - border - 1, 1] ** 2 - x[1] ** 2,
            lambda x: r[zeroP[0] - border, zeroP[1] - border -1, 2] ** 2 - r[
                zeroP[0] - border, zeroP[1] - border -1, 0] ** 2 - x[2] ** 2
        ]

        ga = GA(func=obj_func7, n_dim=1, size_pop=popsize, max_iter=maxiter,
                lb=-r[zeroP[0] - border - 1, zeroP[1] - border - 1, 2],
                ub=r[zeroP[0] - border - 1, zeroP[1] - border - 1, 2]+ prec, precision=prec)
        best_x, _ = ga.run()
        r[zeroP[0] - border - 1, zeroP[1] - border - 1, 1] = best_x[0]
        ga = GA(func=obj_func8, n_dim=3, size_pop=popsize, max_iter=maxiter,
                lb=[-r[zeroP[0] - border - 1, zeroP[1] - border, 2],
                    0,
                    -r[zeroP[0] - border, zeroP[1] - border - 1, 2]],
                ub=[r[zeroP[0] - border - 1, zeroP[1] - border, 2] + prec,
                    r[zeroP[0] - border - 1, zeroP[1] - border - 1, 2] + prec,
                    r[zeroP[0] - border, zeroP[1] - border - 1, 2] + prec],
                precision=[prec, prec, prec], constraint_eq=constraint_eq4)
        best_x, _ = ga.run()
        r[zeroP[0] - border - 1, zeroP[1] - border, 1] = best_x[0]
        r[zeroP[0] - border - 1, zeroP[1] - border - 1, 0] = best_x[1]
        r[zeroP[0] - border, zeroP[1] - border -1, 1] = best_x[2]
        return


    if sec == 0:
        _calR()
        if ky == zeroP[1] + border - 1:
            sec = 1
            print("end 0")
        searching(r, zeroP, border, kx, ky + 1, sec)
    elif sec == 1:
        _calT()
        if kx == zeroP[0] - border + 1:
            sec = 2
            print("end 1")
        searching(r, zeroP, border, kx - 1, ky, sec)
    elif sec == 2:
        _calL()
        if ky == zeroP[1] - border + 1:
            sec = 3
            print("end 2")
        searching(r, zeroP, border, kx, ky - 1, sec)
    elif sec == 3:
        _calD()
        if kx == zeroP[0] + border - 1:
            sec = 4
            print("end 3")
        searching(r, zeroP, border, kx + 1, ky, sec)
    elif sec == 4:
        _calE()
        print("end 4")
        return

def cal_loc(r,zeroP,border, kx, ky, sec):
    def _func(x,y):
        dx = 2*(kx - x)
        dy = 2*(ky - y)
        return r[x,y,5] + r[x,y,0]*dx + r[x,y,1]*dy

    def _calR():
        if ky == zeroP[1] - border:
            r[kx, ky, 5] = _func(kx-1, ky+1)
        else:
            r[kx, ky, 5] = _func(kx-1, ky)

    def _calT():
        if kx == zeroP[1] + border:
            r[kx, ky, 5] = _func(kx-1, ky-1)
        else:
            r[kx, ky, 5] = _func(kx, ky - 1)

    def _calL():
        if ky == zeroP[1] + border:
            r[kx, ky, 5] = _func(kx+1, ky-1)
        else:
            r[kx, ky, 5] = _func(kx + 1, ky)

    def _calD():
        if kx == zeroP[1] - border:
            r[kx, ky, 5] = _func(kx+1, ky+1)
        else:
            r[kx, ky, 5] = _func(kx, ky + 1)

    if sec == 0:
        _calR()
        if ky == zeroP[1] + border - 1:
            sec = 1
            print("end 0")
        cal_loc(r, zeroP, border, kx, ky + 1, sec)
    elif sec == 1:
        _calT()
        if kx == zeroP[0] - border + 1:
            sec = 2
            print("end 1")
        cal_loc(r, zeroP, border, kx - 1, ky, sec)
    elif sec == 2:
        _calL()
        if ky == zeroP[1] - border + 1:
            sec = 3
            print("end 2")
        cal_loc(r, zeroP, border, kx, ky - 1, sec)
    elif sec == 3:
        _calD()
        if kx == zeroP[0] + border - 1:
            return
        cal_loc(r, zeroP, border, kx + 1, ky, sec)

def main(data):
    r = dataHandling(data)
    reconstruct(r)
    for border in range(1, maxborder + 1):
        print('border is', border)
        cal_loc(r, zeroP, border, zeroP[0] + border, zeroP[1] - border, 0)
    draw(r[zeroP[0] - maxborder:zeroP[0] + maxborder + 1, zeroP[1] - maxborder:zeroP[1] + maxborder + 1])

def draw(res):
    fig = plt.figure('3D scatter plot')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(res[:,:, 3], res[:,:, 4], res[:,:, 0], c='r', marker='o')
    plt.show()


if __name__ == "__main__":
    data = importData()
    main(data)
