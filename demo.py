import os
import numpy as np
import re
import json
from pathlib import Path


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


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
                    fvID.append(int(w[0])-1)
                if numUVs > 0:
                    uvID.append(int(w[1])-1)
                if numNormals > 0:
                    nvID.append(int(w[2])-1)
            faceVertIDs.append(fvID)
            uvIDs.append(uvID)
            normalIDs.append(nvID)
            numFaces += 1
    return np.array(vertices), np.array(uvs), np.array(normals), \
        faceVertIDs, uvIDs, normalIDs, vertexColors


def loadObjVerticesAndIndices(filePath):
    vertices, _, _, faceVertIDs, _, _, _ = loadObj(filePath)
    return vertices, faceVertIDs


def loadObjAsDict(filePath, with_faces=True):
    vertices, faceVertIDs = loadObjVerticesAndIndices(filePath)
    name = Path(filePath).stem
    d = {'name': name, 'vertices': vertices.tolist()}
    if with_faces:
        d['vertex_indices'] = faceVertIDs
    return d


def loadObjsAsDict(obj_dir, obj_names, with_faces):
    objs = []
    for obj_name in obj_names:
        obj = loadObjAsDict(os.path.join(obj_dir, obj_name), with_faces)
        objs.append(obj)
    return objs


def saveObj(filePath, vertices, uvs, normals,
            faceVertIDs, uvIDs, normalIDs, vertexColors):
    f_out = open(filePath, 'w')
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" % (len(vertices)))
    f_out.write("# Faces: %s\n" % (len(faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
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


def makeJsonFromObjs(obj_dir, json_path):
    obj_names = [x for x in os.listdir(obj_dir) if x.lower().endswith('.obj')]
    base_name = 'id_0_base.obj'
    identity_names = [x for x in obj_names if x.startswith('id')
                      and x != base_name]
    identity_names = sorted(identity_names, key=natural_keys)
    expression_names = [x for x in obj_names if x.startswith('exp')]
    expression_names = sorted(expression_names, key=natural_keys)
    base = loadObjAsDict(os.path.join(obj_dir, base_name), True)
    identities = loadObjsAsDict(obj_dir, identity_names, False)
    expressions = loadObjsAsDict(obj_dir, expression_names, False)
    d = {'base': base, 'identities': identities, 'expressions': expressions}
    with open(json_path, 'w') as fp:
        json.dump(d, fp)


if __name__ == '__main__':
    eyelid_model_path = './data/eyelid_model.json'
    if not os.path.exists(eyelid_model_path):
        author_eyelid_data_path = './data/EyelidModel/'
        makeJsonFromObjs(author_eyelid_data_path, eyelid_model_path)
    
