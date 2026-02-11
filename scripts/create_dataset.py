#!/usr/bin/env python3

import pandas as pd
import argparse
import sys
import numpy as np
from datetime import datetime, timedelta
import pytz
import yaml
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import math
import os
import cv2
import progressbar

import sequences_definition
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
from pyproj import Proj, Transformer

run = os.system

from dataclasses import dataclass

@dataclass
class CameraInfo:
    K: float = -1.0
    D: float = -1.0
    width: int = -1
    height: int = -1

def compute_direction(lat1, lat2, lon1, lon2):
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)
    return np.array([x, y, bearing])

# Compute associations
def find_correspondences(seq_name, out_path, left_cinfo, lidar_topic, bagfile_path,
                         image_buffer, lidar_buffer, gps_buffer, imu_buffer,
                         yaw_offset, lidar_offset, visualize):

    cmap = plt.colormaps['jet']
    colors = cmap(np.arange(cmap.N))

    row_list_to_dataframe = []

    # LiDAR to Camera extrinsics
    q_corr = Quaternion(axis=[0, 1, 0], angle=yaw_offset)
    q = q_corr * Quaternion([0.701916, 0.711917, -0.0208134, -0.00745561])
    t = np.array([0.1, 0, 0])

    """ Scrim through buffer and look for matches"""
    print(f"Number of SLAM poses: {len(imu_buffer)}")
    print(f"Number of images: {len(image_buffer)}")
    print(f"Number of scans: {len(lidar_buffer)}")
    print(f"Number of gnss_points: {len(gps_buffer)}")

    with AnyReader([Path(bagfile_path)], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:

        progress = create_progressbar()
        progress.start()

        # Pair with imu (TODO: first loop through images, then associate SLAM pose)
        imu_stamp = None
        n_imu_sample = len(imu_buffer)
        for id_imu, imu_data in enumerate(imu_buffer):
            imu_stamp = imu_data[0]

            # Look for closest image
            image_closest_index, image_closest_stamp = min(enumerate(image_buffer), key=lambda x: abs(x[1][0] - imu_stamp))
            if abs(image_closest_stamp[0] - imu_stamp) > 0.1:
                continue

            # Look for closest scan (and seek bagfile for single scan, otherwise memory blows up)
            scan_closest_index, scan_closest_stamp = min(enumerate(lidar_buffer), key=lambda x: abs(x[1][0] - image_closest_stamp[0]))
            if abs(scan_closest_stamp[0] - image_closest_stamp[0]) > 0.3:
                continue
            scan = seek_lidar_message_with_id_in_bagfile(reader, lidar_buffer[scan_closest_index][1],
                                                        lidar_buffer[scan_closest_index][0],
                                                        lidar_topic, lidar_offset)
            # TODO: undistort lidar scan after interpolation of SLAM poses

            # Look for closest GPS
            gps_closest_index, gps_closest_stamp = min(enumerate(gps_buffer), key=lambda x: abs(x[1][4] - imu_stamp))
            if abs(gps_closest_stamp[4] - imu_stamp) > 1.0:
                progress.update(id_imu / n_imu_sample * 100.0)
                continue

            img_lidar_overlay = cv2.imread(image_buffer[image_closest_index][1])
            img_lidar_overlay_orig = cv2.imread(image_buffer[image_closest_index][1])
            points = scan[1]

            pts = list(x[0:3] for i, x in enumerate(points) if  x[3] > 15.0)

            if len(pts) == 0:
                progress.update(id_imu / n_imu_sample * 100.0)
                continue

            trep = np.tile(t, (len(pts), 1))

            pts_cam = np.add(np.dot(q.rotation_matrix, np.transpose(pts)), np.transpose(trep))
            max_dist = np.max(pts_cam[2, :])
            min_dist = np.min(pts_cam[2, :])

            for pt in np.transpose(pts_cam):
                pt_u = left_cinfo.K[0][0] * pt[0] / \
                    pt[2] + left_cinfo.K[0][2]
                pt_v = left_cinfo.K[1][1] * pt[1] / \
                    pt[2] + left_cinfo.K[1][2]
                cmap_pos = np.floor(
                    (pt[2] - min_dist) / max_dist * float(len(colors)))
                if cmap_pos >= len(colors):
                    cmap_pos = len(colors) - 1
                color = np.floor(255 * colors[int(cmap_pos)][0:3])
                cv2.circle(img_lidar_overlay, (int(pt_u), int(
                    pt_v)), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_8)
                cv2.circle(img_lidar_overlay, (int(pt_u), int(pt_v)), 2, (color[2], color[1], color[0]),
                        cv2.FILLED)

            dst = cv2.addWeighted(img_lidar_overlay, .5, img_lidar_overlay_orig,
                                (1 - .5), 0.0)
            cv2.putText(dst, "t_diff: %.3f s" % (abs(image_closest_stamp[0] - scan_closest_stamp[0])),
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(dst, "t_OFF: %.3f s" % lidar_offset, (10, 55), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(dst, "yaw_OFF: %.3f s" % (yaw_offset * 180.0 / np.pi), (10, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
            if dst.all() is not None:
                img_lidar_overlay = dst.copy()

            path_overlay = os.path.join(out_path, "overlays")

            if not os.path.exists(path_overlay):
                os.makedirs(path_overlay)

            path_overlay = os.path.join(
                path_overlay, 'overlay_' + str(image_closest_stamp[0]) + '.png')
            cv2.imwrite(path_overlay, img_lidar_overlay)

            # Get this from the next step
            orientation = np.NAN

            lat = gps_buffer[gps_closest_index][0]
            lon = gps_buffer[gps_closest_index][1]
            elev = gps_buffer[gps_closest_index][-1]

            row_list_to_dataframe.append(
                {'seq_name' : seq_name,
                'time_stamp' : scan_closest_stamp[0],
                'img_path' : image_buffer[image_closest_index][1],
                'overlay' : path_overlay,
                'point_cloud' :  pts_cam.T,
                'latitude' : lat,
                'longitude' : lon,
                'elevation' : elev,
                'orientation' : orientation,
                'slam_orientation' : np.array(imu_data[1]),
                'slam_position' : np.array(imu_data[2])})

            progress.update(id_imu / n_imu_sample * 100.0)

        progress.finish()

    return pd.DataFrame(row_list_to_dataframe, columns=['seq_name',  'time_stamp', 'img_path',
                               'overlay', 'point_cloud', 'latitude', 'longitude', 'elevation', 'orientation',
                               'slam_orientation', 'slam_position'])

def horn(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

# https://github.com/joycesudi/quaternion/blob/main/quaternion.py
def quaternionToEulerAngles(q) -> np.ndarray:
    """
    Convert a quaternion into euler angles [roll, pitch, yaw]
    - roll is rotation around x in radians (CCW)
    - pitch is rotation around y in radians (CCW)
    - yaw is rotation around z in radians (CCW)

    Parameters
    ----------
    q : [4x1] np.ndarray
        quaternion defining a given orientation

    Returns
    -------
    eulerAngles :
        [3x1] np.ndarray
        [roll, pitch, yaw] angles in radians
    """
    if isinstance(q, list) and len(q) == 4:
        q = np.array(q)
    elif isinstance(q, np.ndarray) and q.size == 4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    t2 = 2.0 * (q0 * q2 - q1 * q3)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    if t2 == 1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = -np.arctan2(q0, q1)
    elif t2 == -1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = +np.arctan2(q0, q1)
    else:
        pitch = np.arcsin(t2)
        roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)

    eulerAngles = np.r_[roll, pitch, yaw]

    return eulerAngles

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def compute_length(t):
    diffs = np.sqrt(np.sum(np.diff(t, axis=0)**2, axis=1))
    return np.sum(diffs[diffs<5.0])

def computeRMSE(P, Q):
    if P.shape != Q.shape:
        print("Matrices P and Q must be of the same dimensionality")
        sys.exit(1)
    diff = P - Q
    return np.sqrt((diff**2).mean())

# Assuming positions are already correspondent
def align_slam_gps(alg, gt, skip=1, show_plot=False, dataset_path="", figure_name=""):
    x_gt = gt[0]
    y_gt = gt[1]
    z_gt = gt[2]
    x = alg[0]
    y = alg[1]
    z = alg[2]

    GT = np.column_stack((x_gt, y_gt, z_gt))
    SLAM = np.column_stack((x, y, z))

    R, t = horn(np.transpose(SLAM), np.transpose(GT))
    SLAM_full = np.column_stack((x, y, z))
    SLAM_align = apply_transform(R, t, SLAM)
    SLAM_align_full = apply_transform(R, t, SLAM_full)

    # The "completeness" is defined as the length of the ground truth associated to SLAM estimates
    # versus the length of the full ground truth. Therefore is the perc. of the "travel" covered by SLAM estimates.
    l_gt = compute_length(np.column_stack((x_gt, y_gt, z_gt)))
    l_slam = compute_length(GT)
    print("l_GT: %.2f, l_SLAM: %.2f --> %.2f complete" % (l_gt, l_slam, 100.0 * l_slam / l_gt))

    rmse = computeRMSE(np.transpose(SLAM_align), np.transpose(GT))
    rmse_norm = rmse / l_slam * 100.0
    print("RMSE: %.2f meters (%.2f %%)" % (rmse, rmse_norm))

    SLAM_align_sampled = SLAM_align[::skip]
    GT_align_sampled = GT[::skip]

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for slam_pt, gt_pt in zip(SLAM_align_sampled, GT_align_sampled):
            ax.plot([slam_pt[0], gt_pt[0]], [slam_pt[1], gt_pt[1]], [slam_pt[2], gt_pt[2]], 'r')

        slam_plot, = ax.plot(SLAM_align_full[:, 0], SLAM_align_full[:, 1], SLAM_align_full[:, 2], 'm', label="SLAM")
        gt_plot, = ax.plot(x_gt, y_gt, z_gt, label="GT")

        ax.legend(handles=[slam_plot, gt_plot])
        set_axes_equal(ax)
        ax.view_init(elev=70.)
        plt.savefig(os.path.join(dataset_path, figure_name))
        plt.close()

    if 100.0 * l_slam / l_gt < 10 or rmse_norm > 10:
        return np.NAN, np.NAN, np.array([np.NAN]), np.identity(3)
    else:
        return round(rmse_norm, 2), round(100.0 * l_slam / l_gt, 2), SLAM_align, R

def apply_transform(R, t, points):
    return np.dot(R, points.transpose()).transpose() + t.transpose()

def approximate_northing_from_slam_poses(df, dataset_path, debug=False):
    # 1) Transform the latitude/longitude values into x/y in an ENU frame
    # 2) Align the slam poses to the baseline
    # 2.a) Split slam poses and align independently to better spread the errors
    # 3) Rotate the slam poses and get the angle w.r.t. North

    enu_proj = Proj(proj="aeqd", lat_0=df['latitude'][0], lon_0=df['longitude'][0], ellps="WGS84", datum="WGS84")
    wgs84_proj = Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    transformer = Transformer.from_proj(wgs84_proj, enu_proj)

    enu_coords = np.array([transformer.transform(lon, lat, alt) for lat, lon, alt in
                           zip(df['latitude'].values, df['longitude'].values, df['elevation'].values)])

    n_splits = 3
    gt_splits = np.array_split(enu_coords, n_splits, axis=0)
    slam_positions = [[x, y, z] for x, y, z in df['slam_position'].values]
    slam_splits = np.array_split(np.array(slam_positions), n_splits, axis=0)
    idxs_splits = np.array_split(np.array(range(len(df))), n_splits)
    orientations_northing = []

    for split in range(n_splits):
        print("===== Split {split} ===== ")
        _, _, _, R = align_slam_gps(slam_splits[split].transpose(), gt_splits[split].transpose(),
                                    skip=1, show_plot=debug,
                                    figure_name=df['seq_name'][0] + "_split_" + str(split),
                                    dataset_path=dataset_path)
        q = Quaternion(matrix=R)
        for idx, value in enumerate(df['slam_orientation'][idxs_splits[split]]):

            # Careful here..!! Quaternion must be (w, x, y, z)
            q_enu = q * Quaternion(value)
            euler_enu = quaternionToEulerAngles([q_enu.w, q_enu.x, q_enu.y, q_enu.z])

            # ENU frame points to x-East. To get the angle to the north we must add 90 degrees positive.
            orientations_northing.append(euler_enu[2])

    df['orientation'] = orientations_northing

def create_progressbar():
    widgets = [' [', progressbar.Percentage(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ')']
    return progressbar.ProgressBar(widgets=widgets, maxval=100)

def build_cinfo(filename, target_cam='cam0', opencv_yaml=False):
    """Load a Kalibr or OpenCV yaml file as camera info."""

    cameraInfo = CameraInfo()  # Assuming you are using the class/namespace we discussed

    if opencv_yaml:
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        cameraInfo.K = cv_file.getNode("camera_matrix").mat()
        cameraInfo.D = cv_file.getNode("distortion_coefficients").mat()
        # Add width/height for OpenCV yaml if they exist
        cameraInfo.width = int(cv_file.getNode("image_width").real())
        cameraInfo.height = int(cv_file.getNode("image_height").real())
    else:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)

        cam_data = data.get(target_cam)
        if not cam_data:
            raise ValueError(f"Camera {target_cam} not found in {filename}")

        # 1. Map Resolution
        res = cam_data.get('resolution', [0, 0])
        cameraInfo.width = res[0]
        cameraInfo.height = res[1]

        # 2. Construct K (Camera Matrix)
        # Kalibr format: [fx, fy, cx, cy]
        intrinsics = cam_data.get('intrinsics', [])
        if len(intrinsics) == 4:
            fx, fy, cx, cy = intrinsics
            cameraInfo.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

        # Kalibr Radtan: [k1, k2, p1, p2]
        cameraInfo.D = np.array(cam_data.get('distortion_coeffs', []), dtype=np.float64)

    return cameraInfo

# Get camera info for the target image that we want, with a resize factor applied
# (<1.0) and no distortion
def resize_camera_info_and_rect_maps(src: CameraInfo, resize_factor=1.0): 
    w = src.width
    h = src.height
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(src.K, src.D, (w, h), 1, (w, h))

    new_width = resize_factor * src.width
    new_height = resize_factor * src.height
    scaled_camera_matrix = new_camera_matrix.copy()
    scaled_camera_matrix[0, 0] *= resize_factor
    scaled_camera_matrix[1, 1] *= resize_factor  
    scaled_camera_matrix[0, 2] *= resize_factor  
    scaled_camera_matrix[1, 2] *= resize_factor

    resized = CameraInfo()
    resized.K = scaled_camera_matrix
    resized.width = new_width
    resized.height = new_height
    resized.D = 0.0 * src.D 

    map_u, map_v = cv2.initUndistortRectifyMap(src.K, src.D, None, 
                                             scaled_camera_matrix, (int(new_width), int(new_height)), cv2.CV_32FC1)
                                             
    return resized, map_u, map_v

# With date as string, in format, e.g., "2021/07/08 12:01:55.200"
# Return posix_time
def get_stamp_from_gnss_date(date_string):
    time_format = "%Y/%m/%d %H:%M:%S.%f"
    dt = datetime.strptime(date_string, time_format)
    return dt.timestamp()

# Read GNSS track
# min_quality is 1:fix, 2:float, etc. The lower, the better
def read_track(filename, min_quality=1):
    data = []

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            if tokens[0] == "%":
                continue

            # GNSS time is UTC. For association with "laptop" time, needs conversion to GMT+1, + magic 18 seconds
            stamp = get_stamp_from_gnss_date(tokens[0] + ' ' + tokens[1]) -18 + 7200

            quality = int(tokens[5])
            lat = float(tokens[2])
            lon = float(tokens[3])
            elev = float(tokens[4])

            if quality <= min_quality  :
                data.append([lat, lon, quality, 0.0, float(stamp), elev])

    return data

# Read all .pos files with global measurements from an input folder
# This helps the case where there is not a clear association between sequence and .pos files
def read_all_tracks(path_to_pos_files, min_quality=1):
    data = []

    GPS_EPOCH = datetime(1980, 1, 6)

    for root, dirs, files in os.walk(path_to_pos_files):
        for file in files:
            if file.endswith(".pos"):
                print("Reading from {}".format(file))
                with open(os.path.join(path_to_pos_files, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        tokens = line.split()
                        if tokens[0] == "%":
                            continue

                        # To UTC time
                        gps_week = int(tokens[0])
                        gps_seconds = float(tokens[1])
                        dt_gps = GPS_EPOCH + timedelta(weeks=gps_week, seconds=gps_seconds)

                        # Subtract leap seconds to get UTC time
                        dt_utc = dt_gps - timedelta(seconds=18) + timedelta(hours=2)

                        # UTC to Sicily time zone
                        rome_tz = pytz.timezone('Europe/Rome')
                        dt_rome = dt_utc.astimezone(rome_tz)

                        stamp = dt_rome.timestamp()

                        quality = int(tokens[5])
                        lat = float(tokens[2])
                        lon = float(tokens[3])
                        elev = float(tokens[4])

                        if quality <= min_quality  :
                            data.append([lat, lon, quality, 0.0, float(stamp), elev])

    return data

def parse_pointcloud2_agnostic(msg):
    """
    Parses a sensor_msgs.msg.PointCloud2 message into a structured numpy array.
    Works for any field configuration.
    """
    # Map ROS PointField types to numpy types
    # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointField.html
    # 1: INT8, 2: UINT8, 3: INT16, 4: UINT16, 5: INT32, 6: UINT32, 7: FLOAT32, 8: FLOAT64
    type_map = {
        1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
        5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64
    }

    formats = []
    offsets = []
    for field in msg.fields:
        formats.append(type_map[field.datatype])
        offsets.append(field.offset)

    dtype_dict = {
        'names': [f.name for f in msg.fields],
        'formats': formats,
        'offsets': offsets,
        'itemsize': msg.point_step
    }

    pc_array = np.frombuffer(msg.data, dtype=np.dtype(dtype_dict))
    x = pc_array['x'].astype(np.float32)
    y = pc_array['y'].astype(np.float32)
    z = pc_array['z'].astype(np.float32)
    i = pc_array['intensity'].astype(np.float32)

    return np.column_stack((x, y, z, i))

def seek_lidar_message_with_id_in_bagfile(reader: AnyReader, 
                                          sequence_id, stamp,
                                          lidar_topic, lidar_offset):
    # Convert float stamp (seconds) to nanoseconds for comparison
    t0_ns = int((stamp - 1.0) * 1e9)
    t1_ns = int((stamp + 1.0) * 1e9)

    # Filter connections for the specific lidar topic
    connections = [x for x in reader.connections if x.topic == lidar_topic]

    msg_generator = reader.messages(
        connections=connections,
        start=t0_ns,  # Fast jump starts here
        stop=t1_ns  # Stop reading here
    )

    for connection, timestamp, rawdata in msg_generator:
        msg = reader.deserialize(rawdata, connection.msgtype)

        # Check for sequence ID (if available in your metadata/header)
        if hasattr(msg.header, 'seq') and msg.header.seq == sequence_id:
            msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            time_stamp = msg_time + lidar_offset

            # Use the fast numpy parsing we discussed
            point_cloud = parse_pointcloud2_agnostic(msg)
            return time_stamp, point_cloud

    return None, None

def create_dataset(sequences, path_to_params):

    with open(path_to_params, 'rb') as f:
        params = yaml.safe_load(f.read())

    base_path = params['base_path']
    cinfo_path = base_path + "/" + params['camchain_relative_path']
    resize_factor = params['image_resize_factor']
    debug_mode = params['debug']
    covis_pixel_distance = params['create_dataset']['covisibility_max_pixel_difference']

    yaw_offset = 0
    lidar_offset = 0
    lidar_blend = .5

    left_cinfo = build_cinfo(cinfo_path)
    left_cinfo_resized, map_u, map_v = resize_camera_info_and_rect_maps(left_cinfo, resize_factor)

    print("Reading from sequences: " + str(sequences))

    for seq_name in sequences:

        lidar_buffer = []
        image_buffer = []

        dataset_path = os.path.join(base_path, "dataset", seq_name)
        imu_path = os.path.join(base_path, "processed")

        imgs_dst = os.path.join(dataset_path, "images")
        overlay_dst = os.path.join(dataset_path, "overlays")
        gps_path = os.path.join(base_path, "d-gnss")

        bagfile_path = os.path.join(base_path, "bagfiles", seq_name + ".bag")

        print("Output folder: " + dataset_path)
        print("Reading GPS from {}".format(gps_path))

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        if not os.path.exists(imgs_dst):
            os.makedirs(imgs_dst)

        if not os.path.exists(overlay_dst):
            os.makedirs(overlay_dst)

        bag_path = Path(bagfile_path)
        reader = AnyReader([bag_path])
        reader.open()

        #t0_nanoseconds = reader.start_time + (200 * 1e9)
        #t1_nanoseconds = reader.start_time + (300 * 1e9)

        t0_nanoseconds = reader.start_time
        t1_nanoseconds = reader.end_time

        t0 = t0_nanoseconds * 1e-9
        t1 = t1_nanoseconds * 1e-9

        print("Opened bagfile {} from {} to {}, with duration {}".format(
            str(bagfile_path),
            datetime.fromtimestamp(t0).strftime("%H:%M:%S - %d/%m/%Y"),
            datetime.fromtimestamp(t1).strftime("%H:%M:%S - %d/%m/%Y"),
            str(t1 - t0) + " [s]"))

        left_topic = "/g319c_left/image_raw"
        lidar_topic = "/bf_lidar/points_raw"

        # Read poses from a SLAM algorithm, to get the missing orientation from the ground truth
        df_imu = pd.read_pickle(os.path.join(imu_path, seq_name + "_poses.pkl"))
        print(os.path.join(imu_path, seq_name + "_poses.pkl"))
        print("IMU data shape:", df_imu.shape)
        print(df_imu.head())

        imu_buffer = df_imu[['timestamp', 'rotation', 'position']].to_numpy().tolist()

        # Read GNSS track
        gps_buffer = read_all_tracks(gps_path, 2)
        print("Loaded {} GPS measurements from {} to {}".format(
            len(gps_buffer),
            datetime.fromtimestamp(gps_buffer[0][4]).strftime("%H:%M:%S - %d/%m/%Y"),
            datetime.fromtimestamp(gps_buffer[-1][4]).strftime("%H:%M:%S - %d/%m/%Y")))

        # Dump views of GNSS and SLAM, for debug
        if debug_mode:
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            tmp_poses = np.array([[x[0], x[1]] for x in df_imu['position']])
            tmp_gps = np.array([[x[0], x[1]] for x in gps_buffer])
            axs[0].plot(tmp_poses[:, 0], tmp_poses[:, 1], label="SLAM")
            axs[1].plot(tmp_gps[:, 1], tmp_gps[:, 0], label="GNSS")
            axs[0].axis('equal')
            axs[1].axis('equal')
            fig.tight_layout()
            fig.legend()
            plt.savefig(os.path.join(dataset_path, 'poses_gnss_dump.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print("Reading Images & LiDAR...")

        duration = t1 - t0

        progress = create_progressbar()
        progress.start()

        covis_trigger = ImageCovisibilityTrigger(debug=debug_mode, pixel_median_distance=30.0)

        typestore = get_typestore(Stores.ROS1_NOETIC)

        with AnyReader([Path(bagfile_path)], default_typestore=typestore) as reader:

            connections = [x for x in reader.connections if x.topic in [left_topic, lidar_topic]]

            for connection, timestamp, rawdata in reader.messages(connections=connections,
                                                                  start=t0_nanoseconds, stop=t1_nanoseconds):
                msg = reader.deserialize(rawdata, connection.msgtype)
                topic = connection.topic

                if topic == left_topic:
                    img_l_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                    img_l_color = cv2.cvtColor(img_l_raw, cv2.COLOR_BayerRG2RGB)
                    img_l = cv2.remap(img_l_color, map_u, map_v, cv2.INTER_LINEAR)    

                    #img_l = cv2.resize(img_l_color, (int(left_cinfo_resized.width), int(left_cinfo_resized.height)))

                    if covis_trigger.is_new_sample(img_l):
                        time_stamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

                        path_img = f"{imgs_dst}/img{time_stamp:.6f}.png"
                        cv2.imwrite(path_img, img_l)
                        image_buffer.append([time_stamp, path_img])

                elif topic == lidar_topic:
                    msg_seq = getattr(msg.header, 'seq', 0)
                    time_stamp = (msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)) + lidar_offset
                    lidar_buffer.append([time_stamp, msg_seq])

                progress.update(((float(timestamp)*1e-9 - t0) / duration) * 100)

        progress.finish()

        diff_imu = imu_buffer[-1][0] - imu_buffer[0][0]
        diff_img = image_buffer[-1][0] - image_buffer[0][0]

        print("Delta-t (poses):", diff_imu)
        print("Delta-t (images):", diff_img)

        print("Loaded {} Image measurements from {} to {}".format(
            len(image_buffer),
            datetime.fromtimestamp(image_buffer[0][0]).strftime("%H:%M:%S - %d/%m/%Y"),
            datetime.fromtimestamp(image_buffer[-1][0]).strftime("%H:%M:%S - %d/%m/%Y")))

        print("Loaded {} LiDAR measurements from {} to {}".format(
            len(lidar_buffer),
            datetime.fromtimestamp(lidar_buffer[0][0]).strftime("%H:%M:%S - %d/%m/%Y"),
            datetime.fromtimestamp(lidar_buffer[-1][0]).strftime("%H:%M:%S - %d/%m/%Y")))

        print("Pairing LiDAR and images...")
        df = find_correspondences(seq_name, dataset_path, left_cinfo_resized, lidar_topic, bagfile_path,
                                  image_buffer, lidar_buffer, gps_buffer, imu_buffer, yaw_offset, lidar_offset,
                                  True)
        approximate_northing_from_slam_poses(df, dataset_path, debug_mode)

        # Save to disk
        df.to_pickle(os.path.join(str(dataset_path), str(seq_name) + ".pkl"))
        df.to_csv(os.path.join(str(dataset_path), str(seq_name) + ".csv"))
        df.to_json(os.path.join(str(dataset_path), str(seq_name) + ".json"))
        print("Saving to " + str(os.path.join(str(dataset_path), str(seq_name) + ".pkl")))
        print(df.head)

""" ImageCovisibilityTrigger
    This class implements a trigger, to save image samples, defined by covisibility with previous images. Essentially
    extracts visual keyframes, where the median distance of keypoint matches exceeds an user-defined threshold (defaults
    to 50 pixels). Feature points are extracted on the bottom part of the image, to better reflect changes in the visual
    appearance due to significant motion of the camera. Defaults to ORB detector for speed.. (TODO: implement more)
"""
class ImageCovisibilityTrigger:
    def __init__(self, pixel_median_distance = 50.0, debug=False):
        self.pixel_median_distance = pixel_median_distance
        self.image_reference = None
        self.image_reference_kpts = None
        self.image_reference_desc = None
        self.mask_bottom_half = None
        self.orb = cv2.ORB_create()
        self.hamming_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.debug_mode = debug

    def compute_keypoints_descriptors(self, image):
        if self.mask_bottom_half is None:
            self.mask_bottom_half = np.zeros_like(image)
            height_bottom_half = int(image.shape[0] * 0.5)
            self.mask_bottom_half[-height_bottom_half:] = 255

        return self.orb.detectAndCompute(image, mask=self.mask_bottom_half)

    def is_new_sample(self, image):
        kpts, desc = self.compute_keypoints_descriptors(image)
        
        if not kpts: 
            return True
        
        if self.image_reference is None:
            self.image_reference_kpts = kpts
            self.image_reference_desc = desc
            self.image_reference = image
            return True

        matches = self.hamming_matcher.match(desc, self.image_reference_desc)
        kpts_matched = [kpts[m.queryIdx] for m in matches]
        kpts_reference_matched = [self.image_reference_kpts[m.trainIdx] for m in matches]
        pixel_diffs = [np.sqrt(np.sum(np.array(k_0.pt)-np.array(k_1.pt))**2) for k_0, k_1 in
                       zip(kpts_matched, kpts_reference_matched)]

        if self.debug_mode:
            img_matches = cv2.drawMatches(image, kpts, self.image_reference, self.image_reference_kpts, matches,
                                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.putText(img_matches, "Median: {:.2f}".format(np.median(pixel_diffs)), (50, 30), 1, 1, (255, 0, 0))
            cv2.imshow('Covisibility-based triggering', img_matches)
            cv2.waitKey(10)

        if np.median(pixel_diffs) > self.pixel_median_distance:
            self.image_reference = image
            self.image_reference_kpts = kpts
            self.image_reference_desc = desc
            return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Create S3LI dataset for training and testing')
    
    parser.add_argument('params', type=str, help='path to the config.yaml file with all the parameters')
    #parser.add_argument('path', type=str, help='path to base S3LI folder, as downloaded from the web')
    #parser.add_argument("left_cinfo", type=str,
    #                    help="yaml file with left camera parameters")
    #parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()

    # Load ground truth
    sequences = sequences_definition.sequences

    create_dataset(sequences, args.params)
