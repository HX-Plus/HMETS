import numpy as np

M = 9


# 计算系数
def compute_coefficient(u, v, m=M):
    coefficient = [1, u, v, u * v, u ** 2, v ** 2, u ** 2 * v ** 2, u ** 2 * v, u * v ** 2]
    return coefficient[:m]


# 计算对应视线映射落点
def compute_mapping_gaze(pupil, parameter):
    x, y = 0, 0
    u, v = pupil
    coefficient = compute_coefficient(u, v)
    for i in range(len(coefficient)):
        x += coefficient[i] * parameter[0][i]
        y += coefficient[i] * parameter[1][i]
    return int(x), int(y)


# 计算映射参数
def compute_mapping_parameter(pupils, gazes, m=M):
    parameter = [[0 for i in range(m)] for i in range(2)]

    n = len(pupils)

    if not pupils or not gazes or len(pupils) != len(gazes) or n < m:
        return False, parameter

    coefficients = []
    values_x = []
    values_y = []

    for i in range(n):
        u, v = pupils[i]
        x, y = gazes[i]
        coefficient = compute_coefficient(u, v)
        coefficients.append(coefficient)
        values_x.append([x])
        values_y.append([y])

    coefficients = np.array(coefficients)
    values_x = np.array(values_x)
    values_y = np.array(values_y)
    parameter[0] = np.linalg.solve(coefficients, values_x)
    parameter[1] = np.linalg.solve(coefficients, values_y)

    return True, parameter


def save_mapping_parameter(parameter):
    np.save('./data/mapping_parameter.npy', parameter)
    return True


def load_mapping_parameter():
    try:
        data = np.load('./data/mapping_parameter.npy')
        return True, data
    except FileNotFoundError:
        return False, []

