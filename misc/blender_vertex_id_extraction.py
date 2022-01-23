import bpy
import numpy as np

# Set here as you like
out_dir = 'YOUROUTPUT_PATH'
OBJ_NAME = "base.001"

ob = bpy.data.objects[OBJ_NAME]
group_names = ['r_eye', 'r_upper', 'l_eye', 'l_upper']
group_indices = {}
group_vertices = {}
for group_name in group_names:
    group_vertices[group_name] = []


for group_name in group_names:
    gi = ob.vertex_groups[group_name].index
    indices = []
    for i, v in enumerate(ob.data.vertices):
        for g in v.groups:
            if g.group == gi:
                indices.append(v.index)
    group_indices[group_name] = indices

# X_eye has all eye vertices and X_upper has upper ones
# Make lower ids by set difference
group_indices['r_lower'] = list(set(group_indices['r_eye']).difference(set(group_indices['r_upper'])))
group_indices['l_lower'] = list(set(group_indices['l_eye']).difference(set(group_indices['l_upper'])))


for group_name in group_indices.keys():
    vertices = []
    for idx in group_indices[group_name]:
        for i, v in enumerate(ob.data.vertices):
            if v.index == idx:
                vertices.append(v.co)
            else:
                continue
    group_vertices[group_name] = vertices

# Sort by x
for group_name in ['r_lower', 'r_upper', 'l_lower', 'l_upper']:
    vertices = group_vertices[group_name]
    idx = np.argsort(np.array(vertices), axis=0).T[0]
    indices = group_indices[group_name]
    print(idx, indices)
    swaped_indices = np.take_along_axis(np.array(indices), idx, axis=0)
    print(swaped_indices)
    group_indices[group_name] = swaped_indices


for group_name in group_indices.keys():
    with open(out_dir + group_name + '.txt', 'w', newline='\n') as f:
        print(group_name, len(group_indices[group_name]))
        for index in group_indices[group_name]:
            f.write(str(index)+'\n')
