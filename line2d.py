import torch
import numpy as np
from typing import NamedTuple


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

    return poly(coeffs), coeffs


# Eq. (13) distance precomputation
def calcAccumlatedDists(pixel_pos_list):
    accum_dists = [0.0]
    for i, pix_pos in enumerate(pixel_pos_list):
        if i == 0:
            continue
        d = accum_dists[-1] + torch.linalg.norm(pixel_pos_list[i]
                                                - pixel_pos_list[i-1])
        accum_dists.append(d)
    return accum_dists


# Eq. (13)
def findCorrespondences(projected, pixel_pos_list, line_accum_dists):
    from bisect import bisect_left
    corresp_idx = []
    corresp_pos = []
    proj_accum_dist = calcAccumlatedDists(projected)
    # print(proj_accum_dist)
    # print(pixel_pos_list)
    # print(pixel_pos_list[0])
    for i in range(projected.shape[0]):
        proj_dist = proj_accum_dist[i]
        idx = bisect_left(line_accum_dists, proj_dist)  # TODO
        corresp_idx.append(idx)
        corresp_pos.append(pixel_pos_list[idx])
        #print(pixel_pos_list[idx])
    # print(corresp_pos)
    # print(corresp_idx)
    return corresp_idx, corresp_pos


# https://code.tiblab.net/python/calculate/cubic_equation
def cubic_equation(a, b, c, d):
    p = -b**2/(9.0*a**2) + c/(3.0*a)
    q = b**3/(27.0*a**3) - b*c/(6.0*a**2) + d/(2.0*a)
    t = complex(q**2+p**3)
    w = (-1.0 + 1j*3.0**0.5)/2.0

    u = [0, 0, 0]
    u[0] = (-q + t**0.5)**(1.0/3.0)
    u[1] = u[0] * w
    u[2] = u[0] * w**2
    v = [0, 0, 0]
    v[0] = (-q - t**0.5)**(1.0/3.0)
    v[1] = v[0] * w
    v[2] = v[0] * w**2

    x_list = []
    for i in range(3):
        for j in range(3):
            if abs(u[i]*v[j] + p) < 0.0001:
                x = u[i] + v[j]
                if abs(x.imag) < 0.0000001:
                    x = x.real - b/(3.0*a)
                    x_list.append(x)
    return x_list


def findEyeEndPoints(upper_eq, upper_coeffs, lower_coeffs,
                     approx_r_end, approx_l_end):
    diff = upper_coeffs - lower_coeffs
    a = float(diff[3])
    b = float(diff[2])
    c = float(diff[1])
    d = float(diff[0])
    x_list = cubic_equation(a, b, c, d)
    r_end, r_min, l_end, l_min = None, np.inf, None, np.inf
    for x in x_list:
        y = upper_eq(x)
        pos = np.array([x, y.to('cpu').detach().numpy().copy()])
        r_dist = np.linalg.norm(approx_r_end.to('cpu').detach().numpy().copy() - pos)
        if r_dist < r_min:
            r_min = r_dist
            r_end = pos
        l_dist = np.linalg.norm(approx_l_end.to('cpu').detach().numpy().copy() - pos)
        if l_dist < l_min:
            l_min = l_dist
            l_end = pos
    return r_end, l_end


class EyeInfo(NamedTuple):
    upper_points: torch.tensor
    lower_points: torch.tensor
    upper_accum_dists: list
    lower_accum_dists: list


def createEyeInfo(upper_points_org, upper_w,
                   lower_points_org, lower_w, device):
    upper_curve, upper_coeffs = getCurveFunction(
        upper_points_org, device, upper_w)
    lower_curve, lower_coeffs = getCurveFunction(
        lower_points_org, device, lower_w)
    all_org = torch.cat([upper_points_org, lower_points_org])
    approx_r_end = torch.max(all_org, dim=0)[0]
    approx_r_end = approx_r_end.to(device)
    approx_r_end[1] = upper_curve(approx_r_end[0])
    approx_l_end = torch.min(all_org, dim=0)[0]
    approx_l_end = approx_l_end.to(device)
    approx_l_end[1] = upper_curve(approx_l_end[0])
    r_end, l_end = findEyeEndPoints(upper_curve, upper_coeffs, lower_coeffs,
                                    approx_r_end, approx_l_end)

    upper_points, lower_points = [], []
    # print(approx_r_end, r_end, approx_l_end, l_end)
    for x in range(int(l_end[0]), int(r_end[0])+1):
        upper_points.append([x, upper_curve(x)])
        lower_points.append([x, lower_curve(x)])
    upper_points = torch.tensor(upper_points, device=device)
    lower_points = torch.tensor(lower_points, device=device)
    upper_accum_dists = calcAccumlatedDists(upper_points)
    lower_accum_dists = calcAccumlatedDists(lower_points)
    return EyeInfo(upper_points=upper_points, lower_points=lower_points,
                   upper_accum_dists=upper_accum_dists,
                   lower_accum_dists=lower_accum_dists)


if __name__ == '__main__':
    import cv2
    import numpy as np

    device = torch.device('cuda:0')

    labels = cv2.imread(
        "./data/SemanticEdge/gt_sem_rgb/id_0/result_frame0_l_rgb.png", -1)
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
