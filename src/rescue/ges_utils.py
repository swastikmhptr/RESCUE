import math
import json
import numpy as np
from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation

import plotly.graph_objects as go

def rot_ecef2enu(lat, lon):
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL = np.sin(lamb)
    sP = np.sin(phi)
    cL = np.cos(lamb)
    cP = np.cos(phi)
    rot = np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])
    return rot

def convert_ges_to_mapanything(frames, w, h, ref_frame = 0):
    positions = []
    c2w_list = []
    K_list = []

    lat_ref  = frames[ref_frame]['coordinate']['latitude']
    lon_ref = frames[ref_frame]['coordinate']['longitude']
    alt_ref = frames[ref_frame]['coordinate']['altitude']
    rot = rot_ecef2enu(lat_ref, lon_ref)

    for frame in frames:
        x, y, z = geodetic2enu(
            frame['coordinate']['latitude'],
            frame['coordinate']['longitude'],
            frame['coordinate']['altitude'],
            lat_ref, lon_ref, alt_ref
        )

        positions.append(np.array([x, y, z]))
        rx = frame['rotation']['x']
        ry = frame['rotation']['y']
        rz = frame['rotation']['z']

        R_ecef = Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
        R_enu = rot @ R_ecef

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_enu
        c2w[:3, 3] = np.array([x, y, z])
        c2w_list.append(c2w)

        fl = h / (2 * math.tan(math.radians(frame['fovVertical'] / 2)))
        K = np.array([
            [fl, 0,  w / 2],
            [0,  fl, h / 2],
            [0,  0,  1    ]
        ], dtype=np.float32)
        K_list.append(K)

    return positions, c2w_list, K_list

def convert_ges_to_mapanything_from_file(file_path, ref_frame = 0):
    with open(file_path, 'r') as f:
        data = json.load(f)

    positions, c2w_list, K_list = convert_ges_to_mapanything(data['cameraFrames'], data["width"], data["height"], ref_frame)
    return positions, c2w_list, K_list, data["width"], data["height"]
