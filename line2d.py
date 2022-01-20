import torch

def fitNthOrderEuqation(points, n, device, weights=None):
    X = points[..., 0]  # (D, 1)
    Y = points[..., 1]  # (D, 1)
    if weights is None:
        weights = torch.ones(X.shape)
    A = torch.zeros((n+1, n+1), device=device)
    b = torch.zeros((n+1, 1), device=device)
    X_power = [weights, X * weights]
    for i in range(2, n * 2 + 1):
        X_power.append(X * X_power[-1])

    for i in range(n + 1): 
        b[i] = torch.sum(X_power[i] * Y)

    for j in range(n + 1):
        for i in range(n + 1):
            index = i + j 
            A[i][j] = torch.sum(X_power[index])
            A[j][i] = A[i][j]

    x = torch.linalg.solve(A, b)
    return x


def getCurveFunction(points, device, weights=None):
    coeffs = fitNthOrderEuqation(points, 3, device, weights)

    def polynominal3rd(coeffs, X):
        X2 = X * X
        X3 = X2 * X
        return coeffs[0] + coeffs[1] * X + coeffs[2] * X2 + coeffs[3] * X3

    def poly(coeffs):
        def poly_(X):
            return polynominal3rd(coeffs, X)
        return poly_

    return poly(coeffs)

if __name__ == '__main__':
    import cv2
    import numpy as np

    device = torch.device('cuda:0')

    labels = cv2.imread("./data/SemanticEdge/gt_sem_rgb/id_0/result_frame0_l_rgb.png", -1)
    print(np.count_nonzero(labels), np.where((labels != 0).all(axis=2)))

    bgrLower = np.array([0, 0, 1])
    bgrUpper = np.array([0, 0, 255])
    img_mask = cv2.inRange(labels, bgrLower, bgrUpper)
    red_points = np.where(img_mask == 255)
    weights = torch.tensor(labels[red_points][..., 2])
    red_points = np.stack([red_points[1], red_points[0]]).T
    red_points = torch.tensor(red_points)
    print(red_points)
    curve = getCurveFunction(red_points, device, weights)

    cv2.imwrite("img_mask.png", img_mask)

    for x in range(labels.shape[1]):
        y = curve(x)
        print(x, y)
        center = (int(x), int(y))
        cv2.circle(labels, center, 1, (0, 255, 255), -1)
    cv2.imwrite("curve.png", labels)
