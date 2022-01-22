import os
import numpy as np
import re
import json
from pathlib import Path
import torch
from model import BlendShape


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def loadObj(filePath):
    numVertices = 0
    numUVs = 0
    numNormals = 0
    numFaces = 0
    vertices = []
    uvs = []
    normals = []
    vertexColors = []
    faceVertIDs = []
    uvIDs = []
    normalIDs = []
    for line in open(filePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(x) for x in vals[1:4]]
            vertices.append(v)
            if len(vals) == 7:
                vc = [float(x) for x in vals[4:7]]
                vertexColors.append(vc)
            numVertices += 1
        if vals[0] == "vt":
            vt = [float(x) for x in vals[1:3]]
            uvs.append(vt)
            numUVs += 1
        if vals[0] == "vn":
            vn = [float(x) for x in vals[1:4]]
            normals.append(vn)
            numNormals += 1
        if vals[0] == "f":
            fvID = []
            uvID = []
            nvID = []
            for f in vals[1:]:
                w = f.split("/")
                if numVertices > 0:
                    fvID.append(int(w[0]) - 1)
                if numUVs > 0:
                    uvID.append(int(w[1]) - 1)
                if numNormals > 0:
                    nvID.append(int(w[2]) - 1)
            faceVertIDs.append(fvID)
            uvIDs.append(uvID)
            normalIDs.append(nvID)
            numFaces += 1
    return (
        np.array(vertices),
        np.array(uvs),
        np.array(normals),
        faceVertIDs,
        uvIDs,
        normalIDs,
        vertexColors,
    )


def loadObjVerticesAndIndices(filePath):
    vertices, _, _, faceVertIDs, _, _, _ = loadObj(filePath)
    return vertices, faceVertIDs


def removeUnreferenced(vertices, faces):
    referenced = [False for _ in vertices]
    for f in faces:
        for i in f:
            referenced[i] = True
    vertices_new = []
    count = 0
    table = [-1 for _ in vertices]
    for i, r in enumerate(referenced):
        if r:
            vertices_new.append(vertices[i])
            table[i] = count
            count = count + 1
    faces_new = [f for f in faces]
    print(faces_new)
    for i, f in enumerate(faces):
        faces_new[i][0] = table[faces[i][0]]
        faces_new[i][1] = table[faces[i][1]]
        faces_new[i][2] = table[faces[i][2]]
    return np.array(vertices_new), faces


def loadObjAsDict(filePath, with_faces=True, remove_unreferenced=True):
    vertices, faceVertIDs = loadObjVerticesAndIndices(filePath)
    print(filePath)
    if remove_unreferenced:
        # Original .obj contains unreferenced face vertices
        # Remove them
        vertices, faceVertIDs = removeUnreferenced(vertices, faceVertIDs)
    print(vertices, faceVertIDs)
    name = Path(filePath).stem
    d = {"name": name, "vertices": vertices.tolist()}
    if with_faces:
        d["vertex_indices"] = faceVertIDs
    return d


def loadObjsAsDict(obj_dir, obj_names, with_faces):
    objs = []
    for obj_name in obj_names:
        obj = loadObjAsDict(os.path.join(obj_dir, obj_name), with_faces)
        objs.append(obj)
    return objs


def saveObj(
        filePath, vertices, uvs, normals,
        faceVertIDs, uvIDs, normalIDs, vertexColors, mat_file=None, mat_name=None):
    f_out = open(filePath, "w")
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" % (len(vertices)))
    f_out.write("# Faces: %s\n" % (len(faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
    if mat_file is not None:
        f_out.write("mtllib " + mat_file + "\n")
    for vi, v in enumerate(vertices):
        vStr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" % (color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n" % (len(vertices)))
    for uv in uvs:
        uvStr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvStr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    if mat_name is not None:
        f_out.write("usemtl " + mat_name + "\n")
    for fi, fvID in enumerate(faceVertIDs):
        fStr = "f"
        for fvi, fvIDi in enumerate(fvID):
            fStr += " %s" % (fvIDi + 1)
            if len(uvIDs) > 0:
                fStr += "/%s" % (uvIDs[fi][fvi] + 1)
            if len(normalIDs) > 0:
                fStr += "/%s" % (normalIDs[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(faceVertIDs)))
    f_out.write("# End of File\n")
    f_out.close()


def makeJsonFromObjs(obj_dir, json_path, remove_unreferenced=True):
    obj_names = [x for x in os.listdir(obj_dir) if x.lower().endswith(".obj")]
    base_name = "id_0_base.obj"
    identity_names = [x for x in obj_names if x.startswith(
        "id") and x != base_name]
    identity_names = sorted(identity_names, key=natural_keys)
    expression_names = [x for x in obj_names if x.startswith("exp")]
    expression_names = sorted(expression_names, key=natural_keys)
    base = loadObjAsDict(os.path.join(obj_dir, base_name),
                         True, remove_unreferenced)
    identities = loadObjsAsDict(obj_dir, identity_names,
                                False, remove_unreferenced)
    expressions = loadObjsAsDict(obj_dir, expression_names,
                                 False, remove_unreferenced)
    d = {"base": base, "identities": identities, "expressions": expressions}
    with open(json_path, "w") as fp:
        json.dump(d, fp)


def parseBasises(d):
    vertices = []
    names = []
    for basis in d:
        vertices.append(basis['vertices'])
        names.append(basis['name'])
    return vertices, names


def loadJson(json_path, device):
    with open(json_path, 'r') as fp:
        d = json.load(fp)
    base = torch.tensor(d['base']['vertices'], device=device)
    indices = torch.tensor(d['base']['vertex_indices'],
                           dtype=torch.int32, device=device)
    identities, identity_names = parseBasises(d['identities'])
    identities = torch.tensor(identities, device=device)
    expressions, expression_names = parseBasises(d['expressions'])
    expressions = torch.tensor(expressions, device=device)
    return (base, indices, identities, identity_names,
            expressions, expression_names)


def convertJsonToObjs(json_path, obj_dir):
    base, indices, ids, id_names, exps, exp_names = loadJson(json_path, 'cpu')
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    names = ['base'] + id_names + exp_names
    vertices_list = [base.tolist()] + ids.tolist() + exps.tolist()
    indices = indices.tolist()
    for name, vertices, in zip(names, vertices_list):
        saveObj(
            os.path.join(obj_dir, name+".obj"), vertices, [], [],
            indices, [], [], [])


def loadJsonAsBlendShape(json_path, device) -> BlendShape:
    base, indices, ids, id_names, exps, exp_names = loadJson(json_path, device)
    # print(base.shape, indices.shape, identities.shape, expressions.shape)
    return BlendShape(base, indices, ids, exps)


def _make_ply_txt(pc, color, normal):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(pc)),
                    "property float x", "property float y", "property float z"]
    has_normal = len(pc) == len(normal)
    has_color = len(pc) == len(color)
    if has_normal:
        header_lines += ["property float nx",
                         "property float ny", "property float nz"]
    if has_color:
        header_lines += ["property uchar red", "property uchar green",
                         "property uchar blue", "property uchar alpha"]
    # no face
    header_lines += ["element face 0",
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(pc)):
        line = [pc[i][0], pc[i][1], pc[i][2]]
        if has_normal:
            line += [normal[i][0], normal[i][1], normal[i][2]]
        if has_color:
            line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)
    data_txt = "\n".join(data_lines)

    # no face
    ply_txt = header + data_txt

    return ply_txt


def write_pc_ply_txt(path, pc, color=[], normal=[]):
    with open(path, 'w') as f:
        txt = _make_ply_txt(pc, color, normal)
        f.write(txt)


if __name__ == "__main__":
    if False:
        vertices, _, _, faceVertIDs, _, _, _ = loadObj('./data/cleaned/base.obj')
        with open('r_upper.txt', 'r') as f:
            r_upper_ids = []
            for line in f:
                r_upper_ids.append(int(line.rstrip()))
        r_upper = []
        for i in r_upper_ids:
            print(i)
            r_upper.append(vertices[i])
        write_pc_ply_txt('r_upper.ply', r_upper)

    eyelid_model_path = "./data/eyelid_model.json"
    if not os.path.exists(eyelid_model_path):
        author_eyelid_data_path = "./data/EyelidModel/"
        makeJsonFromObjs(author_eyelid_data_path, eyelid_model_path)
    convertJsonToObjs(eyelid_model_path, './data/cleaned')
    device = torch.device('cuda:0')
    blend_shape = loadJsonAsBlendShape(eyelid_model_path, device)

    from model import OrthoCamera
    identity_coeffs = torch.zeros(
        (blend_shape.identities.shape[0]), device=device)
    expressions_coeffs = torch.zeros(
        (blend_shape.expressions.shape[0]), device=device)
    # print(identity_coeffs, identity_coeffs.shape, bs.identities.shape)
    morphed = blend_shape(identity_coeffs, expressions_coeffs)
    camera = OrthoCamera(device)
    import cv2
    labels = cv2.imread(
        "./data/SemanticEdge/gt_sem_rgb/id_0/result_frame0_l_rgb.png", -1)
    v_max, max_index = torch.max(morphed, dim=0)
    v_min, min_index = torch.min(morphed, dim=0)
    org_diff = v_max - v_min
    v_center = torch.sum(morphed, dim=0) / morphed.shape[0]
    print(labels.shape, v_center)
    h, w, c = labels.shape
    camera.scale = w / org_diff[0]
    # morphed = morphed - v_center
    camera.w2c_t = -v_center * camera.scale + \
        torch.tensor([w/2, h/2, .0], device=device)
    projected = camera(morphed)
    print(projected)
    for p in projected:
        center = (int(p[0]), int(p[1]))
        cv2.circle(labels, center, 1, (255, 255, 255), -1)
    cv2.imwrite("projected.png", labels)
    saveObj("tmp.obj", projected.to('cpu').detach().numpy().copy(
    ), [], [], blend_shape.indices.to('cpu').detach().numpy().copy(), [], [], [])
