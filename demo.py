import mesh_io
import line2d
import model
import torch
import torch.nn as nn
import cv2
import numpy as np
import os

device = torch.device('cuda:0')
parts_vids = {}
parts_vertices = {}
parts_projected = {}
parts_names = ['r_upper', 'r_lower', 'l_upper', 'l_lower']
parts_names_r = ['r_upper', 'r_lower']
parts_names_l = ['l_upper', 'l_lower']


def loadVertexId(path, device):
    with open(path, 'r') as f:
        v_ids = []
        for line in f:
            v_ids.append(int(line.rstrip()))
    return torch.tensor(v_ids, device=device)


def loadPartsVertexIds(base_dir):
    for parts_name in parts_names:
        parts_vids[parts_name] =\
            loadVertexId(os.path.join(base_dir + parts_name + ".txt"), device)


def processSingleEye(labels):
    def process(img, ch):
        bgrLower = np.array([0, 0, 0])
        bgrLower[ch] = 1
        bgrUpper = np.array([0, 0, 0])
        bgrUpper[ch] = 255
        img_mask = cv2.inRange(img, bgrLower, bgrUpper)
        points = np.where(img_mask == 255)
        weights = torch.tensor(img[points][..., ch])
        points = np.stack([points[1], points[0]]).T
        points = torch.tensor(points)
        return points, weights
    red_points, red_weights = process(labels, 2)
    green_points, green_weights = process(labels, 1)
    eye_info = line2d.createEyeInfo(red_points, red_weights,
                                    green_points, green_weights, device)
    return eye_info


if __name__ == '__main__':
    loadPartsVertexIds('./data/cleaned/')

    eyelid_model_path = './data/eyelid_model.json'
    bs = mesh_io.loadJsonAsBlendShape(eyelid_model_path, device)
    identity_coeffs = torch.zeros(
        (bs.identities.shape[0]), device=device)
    expressions_coeffs = torch.zeros(
        (bs.expressions.shape[0]), device=device)

    identity_coeffs = nn.Parameter(identity_coeffs)
    expressions_coeffs = nn.Parameter(expressions_coeffs)

    camera = model.OrthoCamera(device)

    # Initialize RTs of the camera

    # Initialize line features detected in the image
    process_eyes = ['r']  # ['r', 'l']
    labels = cv2.imread(
        "./data/SemanticEdge/gt_sem_rgb/id_0/result_frame0_l_rgb.png", -1)
    eye_infos = {}
    eye_infos['r'] = processSingleEye(labels)

    max_iter = 1000
    optimizer = torch.optim.Adam([identity_coeffs, expressions_coeffs], lr=0.05)
    for i in max_iter:
        optimizer.zero_grad()
        # Morph all vertices
        morphed = bs(identity_coeffs, expressions_coeffs)

        # Extract parts vertices
        for parts_name in parts_names:
            parts_vertices[parts_name] = morphed[parts_vids[parts_name]]
            parts_projected[parts_name] = camera(parts_vertices[parts_name])

        losses = 0.0
        for rl in process_eyes:
            for part in ['_upper', '_lower']:
                # Find correspondences
                part_name = rl + part
                projected = parts_projected[part_name]
                if 'upper' in part:
                    pos_list = eye_infos[rl].upper_points
                    accum_dists = eye_infos[rl].upper_accum_dists
                else:
                    points = eye_infos[rl].lower_points
                    accum_dists = eye_infos[rl].lower_accum_dists
                corresp_idx, corresp_pos = line2d.findCorrespondences
                (projected,
                 points, accum_dists)
                # Take loss
                loss = (projected - corresp_pos) ** 2
                losses = losses + loss
        losses.backward()
        optimizer.step()


    #expressions_coeffs[0] = 1.0
    # print(identity_coeffs, identity_coeffs.shape, bs.identities.shape)
    #bs(identity_coeffs, expressions_coeffs)
    # mesh_io.saveObj("tmp.obj", bs.morphed.to('cpu').detach().numpy().copy(
    # ), [], [], bs.indices.to('cpu').detach().numpy().copy(), [], [], [])
