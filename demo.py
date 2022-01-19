import mesh_io
import line2d
from numpy.core.defchararray import center
import torch
import cv2
import numpy as np

device = torch.device('cuda:0')

if __name__ == '__main__':

    labels = cv2.imread("./data/SemanticEdge/gt_sem_rgb/id_0/result_frame0_l_rgb.png", -1)
    print(np.count_nonzero(labels), np.where((labels != 0).all(axis=2)))

    bgrLower = np.array([0, 0, 1])    # 抽出する色の下限(bgr)
    bgrUpper = np.array([0, 0, 255])    # 抽出する色の上限(bgr)
    img_mask = cv2.inRange(labels, bgrLower, bgrUpper) 
    red_points = np.where(img_mask == 255)
    red_points = np.stack([red_points[1], red_points[0]]).T
    red_points = torch.tensor(red_points)
    print(red_points)
    curve = line2d.getCurveFunction(red_points, device)

    cv2.imwrite("img_mask.png", img_mask)

    for x in range(labels.shape[1]):
        y = curve(x)
        print(x, y)
        center = (int(x), int(y))
        cv2.circle(labels, center, 1, (0, 255, 255), -1)
    cv2.imwrite("curve.png", labels)
    hoge

    eyelid_model_path = './data/eyelid_model.json'

    bs = mesh_io.loadJsonAsBlendShape(eyelid_model_path, device)
    identity_coeffs = torch.zeros(
        (bs.identities.shape[0]), device=device)
    identity_coeffs[0] = 1.0
    expressions_coeffs = torch.zeros(
        (bs.expressions.shape[0]), device=device)
    expressions_coeffs[0] = 1.0
    # print(identity_coeffs, identity_coeffs.shape, bs.identities.shape)
    bs(identity_coeffs, expressions_coeffs)

    mesh_io.saveObj("tmp.obj", bs.morphed.to('cpu').detach().numpy().copy(
    ), [], [], bs.indices.to('cpu').detach().numpy().copy(), [], [], [])
