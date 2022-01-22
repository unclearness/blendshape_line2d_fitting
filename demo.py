from cv2 import HOGDESCRIPTOR_DEFAULT_NLEVELS
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
    print(red_points, red_weights)
    green_points, green_weights = process(labels, 1)
    eye_info = line2d.createEyeInfo(red_points, red_weights,
                                    green_points, green_weights, device)
    return eye_info


def initializeCameraSingleEye(points, vertices, camera):
    v_max, _ = torch.max(vertices, dim=0)
    v_min, _ = torch.min(vertices, dim=0)
    org_diff = v_max - v_min
    v_center = torch.sum(vertices, dim=0) / vertices.shape[0]

    i_max, _ = torch.max(points, dim=0)
    i_min, _ = torch.min(points, dim=0)
    img_diff = i_max - i_min
    i_center = torch.sum(points, dim=0) / points.shape[0]
    camera.scale = img_diff[0] / org_diff[0]
    # print(camera.scale, img_diff, v_center)
    camera.w2c_t = -v_center * camera.scale + \
        torch.tensor([i_center[0], i_center[1], .0], device=device)

    return camera


def drawPoints(img, points, color):
    for i, p in enumerate(points):
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, 1, color, -1)
        #cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.1, color)


def drawEyeUpperLower(img, upper, lower):
    drawPoints(img, upper, (255, 255, 0))
    drawPoints(img, lower, (0, 255, 255))


def drawEyeCorresp(img, src, dst):
    for s, d in zip(src, dst):
        s = (int(s[0]), int(s[1]))
        d = (int(d[0]), int(d[1]))
        cv2.line(img, s, d, (255, 255, 255))


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

    torch.autograd.set_detect_anomaly(True)

    morphed_projected = None

    def update():
        # Morph all vertices
        morphed = bs(identity_coeffs, expressions_coeffs)
        morphed_projected = camera(morphed)
        # Extract parts vertices
        for parts_name in parts_names:
            parts_vertices[parts_name] = morphed[parts_vids[parts_name]]
            parts_projected[parts_name] = camera(parts_vertices[parts_name])
        return morphed, morphed_projected

    # Initialize line features detected in the image
    process_eyes = ['r']  # ['r', 'l']
    labels = cv2.imread(
        "./data/obama_eye_r_label.png", -1)
    labels = labels[..., :3]
    img_name = "obama_eye_r.png"
    img = cv2.imread(
        "./data/" + img_name, -1)
    img = img[..., :3]

    # Resize for better visualization
    # dsize = (300, int(300 * labels.shape[0] / labels.shape[1]))
    # labels = cv2.resize(labels, dsize)
    # img = cv2.resize(img, dsize)

    eye_infos = {}
    eye_infos['r'] = processSingleEye(labels)
    corresp_pos_dict = {}

    out_dir = "./out/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize RTs of the camera
    morphed, morphed_projected = update()
    eye_points = torch.cat([eye_infos['r'].upper_points, eye_infos['r'].lower_points])
    eye_vertices = torch.cat([parts_projected['r_upper'], parts_projected['r_lower']], dim=0)
    camera = initializeCameraSingleEye(eye_points, eye_vertices, camera)
    camera.scale = camera.scale.detach()
    camera.w2c_t = camera.w2c_t.detach()
    camera.scale = nn.Parameter(camera.scale)
    camera.w2c_t = nn.Parameter(camera.w2c_t)
    camera.w2c_q = nn.Parameter(camera.w2c_q)
    max_iter = 1000
    optimizer = torch.optim.Adam([identity_coeffs, expressions_coeffs, camera.scale, camera.w2c_t, camera.w2c_q], lr=0.005)
    for i in range(max_iter):
        optimizer.zero_grad()

        morphed, morphed_projected = update()

        losses = 0.0
        for rl in process_eyes:
            for part in ['_upper', '_lower']:
                # Find correspondences
                part_name = rl + part
                projected = parts_projected[part_name]
                projected = projected[..., :2]  # ignore Z
                if 'upper' in part:
                    points = eye_infos[rl].upper_points
                    accum_dists = eye_infos[rl].upper_accum_dists
                else:
                    points = eye_infos[rl].lower_points
                    accum_dists = eye_infos[rl].lower_accum_dists
                # print(points)
                corresp_idx, corresp_pos = line2d.findCorrespondences(projected,
                                                                      points, accum_dists)
                # Take loss
                corresp_pos = torch.stack(corresp_pos, dim=0)
                corresp_pos_dict[part_name] = corresp_pos
                loss = torch.sum(torch.abs(projected - corresp_pos))
                # print(loss)
                losses = losses + loss
        # w = 1000.0
        # losses += torch.sum(- expressions_coeffs[expressions_coeffs < 0]) * w
        # losses += torch.sum(- identity_coeffs[identity_coeffs < 0]) * w
        # with torch.no_grad():
        #     expressions_coeffs[:] = torch.clamp(expressions_coeffs, 0.0, 1.0)
        #     identity_coeffs[:] = torch.clamp(identity_coeffs, 0.0, 1.0)
        losses.backward(retain_graph = True)
        optimizer.step()
        if i % 1 == 0:
            print(i)
            print(losses)
            print(expressions_coeffs)
            print(identity_coeffs)
            tmp = labels.copy()
            drawEyeCorresp(tmp, parts_projected['r_upper'], corresp_pos_dict['r_upper'])
            drawEyeCorresp(tmp, parts_projected['r_lower'], corresp_pos_dict['r_lower'])
            drawEyeUpperLower(tmp, parts_projected['r_upper'], parts_projected['r_lower'])
            cv2.imwrite(out_dir + "label_" + str(i)+'.png', tmp)
            tmp = img.copy()
            drawEyeCorresp(tmp, parts_projected['r_upper'], corresp_pos_dict['r_upper'])
            drawEyeCorresp(tmp, parts_projected['r_lower'], corresp_pos_dict['r_lower'])
            drawEyeUpperLower(tmp, parts_projected['r_upper'], parts_projected['r_lower'])
            cv2.imwrite(out_dir + "img_" + str(i) +'.png', tmp)
        if i % 100 == 0:
            mesh_io.saveObj(out_dir + "eye_" + str(i) + ".obj", morphed_projected.to('cpu').detach().numpy().copy(),
             [], [], bs.indices.to('cpu').detach().numpy().copy(), [], [], [])
            
            z = 0.0
            tmp_v = np.array([[0.0, 0.0, z], [float(labels.shape[1]), 0.0, z],
             [0.0, float(labels.shape[0]), z], [float(labels.shape[1]), float(labels.shape[0]), z]])
            tmp_uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            tmp_idx = [[0, 1, 2], [1, 3, 2]]

            mat_name = img_name[:-4]
            mesh_io.saveObj(out_dir + "input_" + str(i) + ".obj", tmp_v,
            tmp_uv, [], tmp_idx, tmp_idx, [], [], mat_file="input.mtl", mat_name=mat_name)

            with open(out_dir + "input.mtl", 'w') as f:
                f.write(
                f"""newmtl {mat_name}
                Ka 0.117647 0.117647 0.117647
                Kd 0.752941 0.752941 0.752941
                Ks 0.752941 0.752941 0.752941
                Tr 0
                illum 1
                Ns 8
                map_Kd """ + img_name)
            cv2.imwrite(out_dir + img_name, img)

    #expressions_coeffs[0] = 1.0
    # print(identity_coeffs, identity_coeffs.shape, bs.identities.shape)
    #bs(identity_coeffs, expressions_coeffs)
    # mesh_io.saveObj("tmp.obj", bs.morphed.to('cpu').detach().numpy().copy(
    # ), [], [], bs.indices.to('cpu').detach().numpy().copy(), [], [], [])
