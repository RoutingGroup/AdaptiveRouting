'''
Calculate the PMI

Author: Liu Baoju
Date: 2023/11/16
'''

import numpy as np


def calcu_theta(N, m, x, u):
    temp = 0
    for j in range(N[m]):
        temp += (x[j][m] - u[m]) ** 2
    return 1 / N[m] * temp


def calcu_alpha(C, i, m, m_total):
    num = C[i][m]
    denom = 0
    for m_idx in range(m_total):
        denom += C[i][m_idx]
    return num / denom


'''
The probability of choosing the m-th routing objective in the first stage
'''
def probability_m(m,m_total, C, x, u, N, V):
    def num():
        num = 0
        for j in range(N[m]):
            alpha_j_m = calcu_alpha(C, j, m, m_total)
            theta_m = calcu_theta(N, m, x, u)
            num += (alpha_j_m * np.exp(V[j])) ** (1 / theta_m)
        return num

    def denom():
        denom = 0
        for k in range(m):
            temp = 0
            for i in range(N[m]):
                alpha_i_m = calcu_alpha(C, i, m, m_total)
                theta_m = calcu_theta(N, m, x, u)
                temp += (alpha_i_m * np.exp(V[i])) ** (1 / theta_m)
            theta_k = calcu_theta(N, k, x, u)
            denom += temp ** (1 / theta_k)
        return denom

    return num() / denom()


'''
The conditional probability of choosing the i-th type of shortest path in the second stage
'''
def probability_i_m(m, m_total, C, x, u, N, V, i):
    def num():
        alpha_i_m = calcu_alpha(C, i, m, m_total)
        theta_m = calcu_theta(N, m, x, u)
        return (alpha_i_m * np.exp(V[i])) ** (1 / theta_m)

    def denom():
        denom = 0
        for j in range(N[m]):
            alpha_j_m = calcu_alpha(C, j, m, m_total)
            theta_m = calcu_theta(N, m, x, u)
            denom += (alpha_j_m * np.exp(V[j])) ** (1 / theta_m)
        return denom

    return num() / denom()


'''
Calculate the PMI for R_i
'''
def PMI_i(m_total, C, x, u, N, V, i):
    PMI_i_val = 0
    for m in range(m_total):
        prob_m = probability_m(m,m_total, C, x, u, N, V)
        prob_i_m = probability_i_m(m, m_total, C, x, u, N, V, i)
        PMI_i_val += prob_m * prob_i_m
    return PMI_i_val


'''
Calculate the path selection probability of PMI

:param C: the deviation rate between the normalized travel, for example: C_i_m denotes the cost path R_i and the minimum normalized cost inthe m-th routing objective
:param x: the normalized travel cost of path,for example: x_i_m denotes the normalized travel cost of path R_i in the m-th routing objective
:param u: the  mean of the normalized travel cost, for example: u_m denotes the mean of the normalized travel cost in the m-th routing objective
:param N: the numbers of paths, for example: N_m are the numbers of paths in the m-th routing objectives,
:param V: the deterministic component of U_i, which is measured via composite similarity SIM.
:param m_total: the number of nodes in each path, The recommended number is 5

returns: PMI type: list
'''
def PMI(C, x, u, N, V, m_total=5):
    PMI = []
    for i in range(m_total):
        PMI.append(PMI_i(m_total, C, x, u, N, V, i))
    return PMI

