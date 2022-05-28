import numpy as np


def compute_mapping_matrix(imgA_size, imgB_size, pupils, gazes):
    n = len(pupils)

    if not pupils or not gazes or len(pupils) != len(gazes) or n != 9:
        return False, []

    mA, nA = imgA_size[1], imgA_size[0]
    mB, nB = imgB_size[1], imgB_size[0]

    mapping_matrix = [[[0, 0] for i in range(mA)] for i in range(nA)]
    mapping_matrix = np.asarray(mapping_matrix)

    # index = np.lexsort([pupils[:, 1], pupils[:, 0]])
    index = [i for i in range(n)]

    pupil_map = [
        [[0, 0], [pupils[index[0]][0], 0], [pupils[index[1]][0], 0], [pupils[index[2]][0], 0], [nA - 1, 0]],
        [[0, pupils[index[0]][1]], pupils[index[0]], pupils[index[1]], pupils[index[2]], [nA - 1, pupils[index[2]][1]]],
        [[0, pupils[index[3]][1]], pupils[index[3]], pupils[index[4]], pupils[index[5]], [nA - 1, pupils[index[5]][1]]],
        [[0, pupils[index[6]][1]], pupils[index[6]], pupils[index[7]], pupils[index[8]], [nA - 1, pupils[index[8]][1]]],
        [[0, mA - 1], [pupils[index[6]][0], mA - 1], [pupils[index[7]][0], mA - 1], [pupils[index[8]][0], mA - 1],
         [nA - 1, mA - 1]]
    ]

    gaze_map = [
        [[0, 0], [gazes[index[0]][0], 0], [gazes[index[1]][0], 0], [gazes[index[2]][0], 0], [nB - 1, 0]],
        [[0, gazes[index[0]][1]], gazes[index[0]], gazes[index[1]], gazes[index[2]], [nB - 1, gazes[index[2]][1]]],
        [[0, gazes[index[3]][1]], gazes[index[3]], gazes[index[4]], gazes[index[5]], [nB - 1, gazes[index[5]][1]]],
        [[0, gazes[index[6]][1]], gazes[index[6]], gazes[index[7]], gazes[index[8]], [nB - 1, gazes[index[8]][1]]],
        [[0, mB - 1], [gazes[index[6]][0], mB - 1], [gazes[index[7]][0], mB - 1], [gazes[index[8]][0], mB - 1],
         [nB - 1, mB - 1]]
    ]

    for i in range(len(pupil_map) - 1):
        for j in range(len(pupil_map[0]) - 1):
            pul, pur, pbl, pbr = pupil_map[i][j], pupil_map[i][j + 1], pupil_map[i + 1][j], pupil_map[i + 1][j + 1]
            gul, gur, gbl, gbr = gaze_map[i][j], gaze_map[i][j + 1], gaze_map[i + 1][j], gaze_map[i + 1][j + 1]
            compute_mapping_part_iteration([pul, pur, pbl, pbr], [gul, gur, gbl, gbr], mapping_matrix)

    return True, mapping_matrix


def compute_mapping_part(ps, gs, matrix):
    pul, pur, pbl, pbr = ps
    gul, gur, gbl, gbr = gs

    if pbr[0] - pul[0] <= 1 or pbr[1] - pul[1] <= 1:
        return

    matrix[pul[0]][pur[1]] = gul
    matrix[pur[0]][pur[1]] = gur
    matrix[pbl[0]][pbl[1]] = gbl
    matrix[pbr[0]][pbr[1]] = gbr

    pu = [int((pul[0] + pur[0]) / 2), int((pul[1] + pur[1]) / 2)]
    gu = [int((gul[0] + gur[0]) / 2), int((gul[1] + gur[1]) / 2)]
    pl = [int((pul[0] + pbl[0]) / 2), int((pul[1] + pbl[1]) / 2)]
    gl = [int((gul[0] + gbl[0]) / 2), int((gul[1] + gbl[1]) / 2)]
    pr = [int((pur[0] + pbr[0]) / 2), int((pur[1] + pbr[1]) / 2)]
    gr = [int((gur[0] + gbr[0]) / 2), int((gur[1] + gbr[1]) / 2)]
    pb = [int((pbl[0] + pbr[0]) / 2), int((pbl[1] + pbr[1]) / 2)]
    gb = [int((gbl[0] + gbr[0]) / 2), int((gbl[1] + gbr[1]) / 2)]

    pm = [int((pl[0] + pr[0]) / 2), int((pu[1] + pb[1]) / 2)]
    gm = [int((gl[0] + gr[0]) / 2), int((gu[1] + gb[1]) / 2)]

    compute_mapping_part([pul, pu, pl, pm], [gul, gu, gl, gm], matrix)
    compute_mapping_part([pu, pur, pm, pr], [gu, gur, gm, gr], matrix)
    compute_mapping_part([pl, pm, pbl, pb], [gl, gm, gbl, gb], matrix)
    compute_mapping_part([pm, pr, pb, pbr], [gm, gr, gb, gbr], matrix)


def compute_mapping_part_iteration(ps, gs, matrix):
    queue = [[ps, gs]]
    while len(queue):
        ps, gs = queue[0]
        queue.pop(0)

        pul, pur, pbl, pbr = ps
        gul, gur, gbl, gbr = gs

        matrix[pul[0]][pur[1]] = gul
        matrix[pur[0]][pur[1]] = gur
        matrix[pbl[0]][pbl[1]] = gbl
        matrix[pbr[0]][pbr[1]] = gbr

        if (pur[0] - pul[0] <= 1 or pbr[0] - pbl[0] <= 1) and (pbl[1] - pul[1] <= 1 or pbr[1] - pur[1] <= 1):
            continue

        pu = [int((pul[0] + pur[0]) / 2), int((pul[1] + pur[1]) / 2)]
        gu = [int((gul[0] + gur[0]) / 2), int((gul[1] + gur[1]) / 2)]
        pl = [int((pul[0] + pbl[0]) / 2), int((pul[1] + pbl[1]) / 2)]
        gl = [int((gul[0] + gbl[0]) / 2), int((gul[1] + gbl[1]) / 2)]
        pr = [int((pur[0] + pbr[0]) / 2), int((pur[1] + pbr[1]) / 2)]
        gr = [int((gur[0] + gbr[0]) / 2), int((gur[1] + gbr[1]) / 2)]
        pb = [int((pbl[0] + pbr[0]) / 2), int((pbl[1] + pbr[1]) / 2)]
        gb = [int((gbl[0] + gbr[0]) / 2), int((gbl[1] + gbr[1]) / 2)]

        pm = [int((pl[0] + pr[0]) / 2), int((pu[1] + pb[1]) / 2)]
        gm = [int((gl[0] + gr[0]) / 2), int((gu[1] + gb[1]) / 2)]

        queue.append([[pul, pu, pl, pm], [gul, gu, gl, gm]])
        queue.append([[pu, pur, pm, pr], [gu, gur, gm, gr]])
        queue.append([[pl, pm, pbl, pb], [gl, gm, gbl, gb]])
        queue.append([[pm, pr, pb, pbr], [gm, gr, gb, gbr]])


def save_mapping_matrix(matrix):
    np.savez('./data/mapping_matrix.npz', matrix)
    return True


def load_mapping_matrix():
    try:
        data = np.load('./data/mapping_matrix.npz')
        return True, data[data.files[0]]
    except FileNotFoundError:
        return False, []
