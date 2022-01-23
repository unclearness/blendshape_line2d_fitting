# Blendshape line fitting with pytorch
Fit 3D lines define on the blendshape mesh to 2D lines detected on the image.
In particular, this code tries to fit eye lines.
Optimization is performed by a gradient based method with pytorch.
This is a very partial implementation of "Real-time 3D Eyelids Tracking from Semantic Edges"

<img src="https://user-images.githubusercontent.com/1129855/150648942-fe84579f-2dbc-4976-b200-69f115473751.png" width="480">

<img src="https://user-images.githubusercontent.com/1129855/150648048-029a5f3f-3871-4507-b1e3-4a9ae4540f29.gif" width="480">



# Run
- Download `EyelidModel` from the authors' site and unzip `./data/`
- `python demo.py`
- Check images and objs under `./out/`
- `python make_gif.py` if you want to make gif

# Difference from the paper
- Pytorch based optimization.
- Only two eye lines are fitted.
  - The paper fits four lines.
- Blendshape range is not limited. (typically [0, 1] in practice)
- Not very accurate because of the above reasons
  - The two lines are fitted but the algorithm does not care about the other parts.
- More differences...

|Initial|Optimized|
|---|---|
|![image](https://user-images.githubusercontent.com/1129855/150648453-63dd31c1-024f-46ea-9cdb-04b82be18849.png)|![image](https://user-images.githubusercontent.com/1129855/150648465-2670ad1f-a81b-42d4-a056-9473995cdae4.png)|


# Dependencies
- pytorch
- numpy
- cv2

# Misc
The original model includes unreferenced vertices (non eye parts, such as head). The code remove them first. Then, cleaned models will be generated in `./data/cleaned/`. Attached files (e.g., `r_lower.txt`) describe vertex ids of the corresponding eye parts for the cleanded models. I made the files by `./misc/blender_vertex_id_extraction.py` with Blender 2.92. I manually selected vertex groups on the cleaned models and run the script.

# Reference
- Quan Wen, Feng Xu, Ming Lu and Jun-Hai Yong. 2017. "Real-time 3D Eyelids Tracking from Semantic Edges". ACM Transaction on Graphics (TOG).
  - http://xufeng.site/projects/realtime_3d_eyelids/index.html


