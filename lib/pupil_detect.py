import cv2
import numpy as np

FACTOR_GRADIENT_THRESHOLD = 50.0
FACTOR_BLUR_SIZE = 5
FACTOR_WEIGHT = 1.0
FACTOR_DOT_PRODUCT_THRESHOLD_RATIO = 0.6
FACTOR_DILATE = 10
FACTOR_ERODE = 8

displacement_X, displacement_Y = None, None
displacement_w, displacement_h = 320, 180


# 计算梯度阀值
def compute_threshold(matrix, factor):
    rows, cols = matrix.shape
    mean, std = cv2.meanStdDev(matrix)
    stddev = std[0] / (rows * cols) ** 0.5
    threshold = factor * stddev + mean[0]
    return threshold


# 过滤低于阀值的梯度
def process_gradient(gradient_X, gradient_Y, magnitude):
    threshold = compute_threshold(magnitude, FACTOR_GRADIENT_THRESHOLD)
    gradient_X = np.where(magnitude < threshold, 0, gradient_X)
    gradient_Y = np.where(magnitude < threshold, 0, gradient_Y)
    gradient_X = np.divide(gradient_X, magnitude, out=np.zeros_like(gradient_X), where=magnitude != 0)
    gradient_Y = np.divide(gradient_Y, magnitude, out=np.zeros_like(gradient_Y), where=magnitude != 0)
    return gradient_X, gradient_Y, magnitude


# 反转灰度
def reverse_grayscale(matrix):
    return 255 - matrix


# 计算点积和
def compute_dot_product(gradient_X, gradient_Y, weight_arr, weight_factor=FACTOR_WEIGHT):
    rows, cols = weight_arr.shape
    dot_product_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            gX = gradient_X[i][j]
            gY = gradient_Y[i][j]
            if gX == 0 and gY == 0:
                continue
            for row in range(rows):
                for col in range(cols):
                    if i == row and j == col:
                        continue
                    dx = j - col
                    dy = i - row
                    m = (dx ** 2 + dy ** 2) ** 0.5
                    dx /= m
                    dy /= m
                    dot_product = dx * gX + dy * gY
                    dot_product = max(0.0, dot_product)
                    dot_product_matrix[row][col] += dot_product ** 2 * weight_arr[row][col] * weight_factor
    return dot_product_matrix


def fast_dot_product(gradient_X, gradient_Y, weight_arr, displacement_X, displacement_Y, weight_factor=FACTOR_WEIGHT):
    x = cv2.filter2D(gradient_X, -1, displacement_X, borderType=cv2.BORDER_CONSTANT)
    y = cv2.filter2D(gradient_Y, -1, displacement_Y, borderType=cv2.BORDER_CONSTANT)
    m = x + y
    ret, m = cv2.threshold(m, 0, 0, cv2.THRESH_TOZERO)
    dot_product_matrix = m * weight_arr * weight_factor
    return dot_product_matrix


def compute_displacement(w, h):
    rows = 2 * h + 1
    cols = 2 * w + 1
    displacement_X = np.zeros((rows, cols))
    displacement_Y = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            x = j - w
            y = i - h
            m = np.hypot(x, y)
            if m == 0:
                continue
            x /= m
            y /= m
            displacement_X[i][j] = x
            displacement_Y[i][j] = y
    return displacement_X, displacement_Y


def load_displacement(w=displacement_w, h=displacement_h):
    try:
        data = np.load('./data/displacement_parameter.npz')
        displacement_X = data[data.files[0]]
        displacement_Y = data[data.files[1]]
        if displacement_X.shape[0] < h or displacement_X.shape[0] < w:
            displacement_X, displacement_Y = compute_displacement(w, h)
            np.savez('./data/displacement_parameter.npz', displacement_X, displacement_Y)
    except FileNotFoundError:
        displacement_X, displacement_Y = compute_displacement(w, h)
        np.savez('./data/displacement_parameter.npz', displacement_X, displacement_Y)
    return displacement_X, displacement_Y


# 过滤边缘
def post_processing(dot_product_matrix, thresh):
    rows, cols = dot_product_matrix.shape
    res = dot_product_matrix.copy()
    visited = np.zeros((rows, cols))
    list = []
    for i in range(rows):
        list.append((i, 0))
        visited[i][0] = 1
    for j in range(cols):
        list.append((0, j))
        visited[0][j] = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while len(list) != 0:
        px, py = list.pop()
        dot_product_matrix[px][py] = 0
        for direction in directions:
            tx = px + direction[0]
            ty = py + direction[1]
            if 0 <= tx < rows and 0 <= ty < cols and visited[tx][ty] == 0 and dot_product_matrix[tx][ty] > thresh:
                list.append((tx, ty))
                visited[tx][ty] = 1
    return res


# 寻找质心
def find_centroid(img, factor_dilate=FACTOR_DILATE, factor_erode=FACTOR_ERODE, img_list=[]):
    kernel = np.zeros((9, 9), dtype="uint8")
    kernel = cv2.circle(kernel, (4, 4), 4, (255, 255, 255), -1)
    img = cv2.dilate(img, kernel, iterations=factor_dilate)
    # show_img(img, "2")
    img_list.append(process_img(img.copy(), v=255))
    img = cv2.erode(img, kernel, iterations=factor_erode)
    # show_img(img, "3")
    img_list.append(process_img(img.copy(), v=255))

    rows, cols = img.shape
    x = y = wx = wy = 0
    weight = np.sum(img)
    for i in range(rows):
        wy += np.sum(img[i, :])
        if wy >= weight / 2:
            y = i
            break
    for i in range(cols):
        wx += np.sum(img[:, i])
        if wx >= weight / 2:
            x = i
            break
    return x, y


# 瞳孔检测
def detect_pupil(
        img, w=100,
        factor_threshold_ratio=FACTOR_DOT_PRODUCT_THRESHOLD_RATIO,
        factor_dilate=FACTOR_DILATE,
        factor_erode=FACTOR_ERODE,
        img_list=[]
):
    k = img.shape[0] / w if img.shape[0] / w >= 1 else 1
    h = int(img.shape[1] / k)
    img = cv2.resize(img, (h, w))

    global displacement_X
    global displacement_Y

    if displacement_X is None or displacement_Y is None:
        dis_w = max(displacement_w, w)
        dis_h = max(displacement_h, h)
        displacement_X, displacement_Y = load_displacement(dis_w, dis_h)

    dis_h, dis_w = displacement_X.shape

    dis_X = displacement_X[
            dis_h - h: dis_h + h + 1,
            dis_w - w: dis_w + w + 1
            ]
    dis_Y = displacement_Y[
            dis_h - h: dis_h + h + 1,
            dis_w - w: dis_w + w + 1
            ]

    img = cv2.GaussianBlur(img, (FACTOR_BLUR_SIZE, FACTOR_BLUR_SIZE), sigmaX=0, sigmaY=0)
    img_arr = np.asarray(img)
    weight_arr = reverse_grayscale(img_arr)

    gradient_Y, gradient_X = np.gradient(img_arr)
    magnitude = np.hypot(gradient_Y, gradient_X)

    gradient_X, gradient_Y, magnitude = process_gradient(gradient_X, gradient_Y, magnitude)

    # dot_product_matrix = compute_dot_product(gradient_X.copy(), gradient_Y.copy(), weight_arr.copy())
    dot_product_matrix = fast_dot_product(gradient_X, gradient_Y, weight_arr, dis_X, dis_Y)

    # show_img(dot_product_matrix, "0")

    _, max_val, _, max_p = cv2.minMaxLoc(dot_product_matrix)

    # dot_product_matrix = dot_product_matrix.astype(np.uint8)

    ret, res = cv2.threshold(dot_product_matrix, max_val * factor_threshold_ratio, 0, cv2.THRESH_TOZERO)

    # show_img(res, "1")
    img_list.append(process_img(res.copy(), v=255))

    _, max_val, _, max_p = cv2.minMaxLoc(res)

    # location = max_p

    location = find_centroid(res, factor_dilate, factor_erode, img_list)
    location = (round(location[0] * k), round(location[1] * k))

    return location


# 处理图片
def process_img(img, v=0.5, k=1):
    _, max_val, _, _ = cv2.minMaxLoc(img)
    img = img / max_val * v
    img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
    return img


# 显示图片
def show_img(img, name, k=1):
    img = process_img(img, k)
    cv2.imshow(name, img)
    return img
