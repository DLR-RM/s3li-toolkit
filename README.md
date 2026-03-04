# S3LI Interface
This package provides useful tools and scripts for playback and preparation of the S3LI dataset. 

This repo relates to the **S3LI Vulcano** release, for the Etna dataset, go back to the main branch. 

![bokeh_plot](doc/s3li_banner.gif)

[![arXiv](https://img.shields.io/badge/arXiv-2601.19557-b31b1b.svg)](https://arxiv.org/abs/2601.19557)
[![Data](https://img.shields.io/badge/Data-S3LI--Vulcano-green?logo=databricks&logoColor=white)](https://rmc.dlr.de/s3li_dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## ROS2 Interface 
We provide a "starter" interface to play the data back, and inspect the bagfile content. 
The `s3li_ros` folder is a ROS2 package that provides a launchfile to execute the following components: 
- `calde_to_camerainfo_ros.py`: a node that reads camera calibration files (in DLR CalDe/CalLab format) and republishes camera_info messages 
- `stereo_disparity_node`: subscribes to the raw images and camera info messages, performs stereo rectification and publishes depth images. Optionally (recommended), the node first downsamples the input images with a custom factor ($(u´, v´) = f * (u, v)$, with f preset to 0.25)
- static tf publishers for the following transformations: 
  - `camera_left -> camera_right`
  - `camera_left -> lidar`
  - `imu -> camera_left`

LiDAR messages are provide as recorded from the sensor. Published messages, as `PointCloud2` comprise the following point fields: 
```
N° Points:  17400
Point Step: 28 bytes

FIELDS:
 - Name: x                 | Offset: 0   | Datatype: float32 
 - Name: y                 | Offset: 4   | Datatype: float32
 - Name: z                 | Offset: 8   | Datatype: float32  
 - Name: intensity         | Offset: 12  | Datatype: uint32  
 - Name: ambient_light     | Offset: 16  | Datatype: uint32  
 - Name: point_id          | Offset: 20  | Datatype: uint32  
 - Name: point_time_offset | Offset: 24  | Datatype: uint32  
```
Spurious measurements can be filtered out setting thresholds on the `intensity` field. As an example check the `inspect_pt_and_filter.py` script.
LiDAR messages are PTP synchronized with the clock of the main pc, to which cameras are synchronized too with the same mechanism. The XSens-IMU, connected via USB, however, is not. To properly synchronize therefore LiDAR messages with IMU readings, consider to rely on the time offset of `0.009` seconds estimated between camera and IMU messages with Kalibr (files in the data folder). 

To use the ROS2 interface, first convert the bagfiles: 
```
pip install rosbags
rosbags-convert ${input_ros1_bag}
```

then clone and build the `s3li-toolkit` and its `s3li_ros2` package: 
```
mkdir -p ~/s3li_ws/src
cd ~/s3li_ws/src
git clone https://github.com/DLR-RM/s3li-toolkit.git
cd ..
colcon build --packages-select s3li_ros2
source install/setup.bash 
ros2 launch s3li_ros2 s3li_node.launch.py
```
## Generation of train and test datasets for place recognition
The package provides two scripts to scrim the bagfiles and generate datasets with synchronized tuples of:
$$(I_{left}, L, p_{D-GNSS}, \phi_{north})$$
with $I_{left}$ the left camera image, $L$ a lidar scan, $p_{D-GNSS}$ the ground truth position in global coordinates, measured with a differential GNSS setup, and $\phi_{north}$ the orientation of the camera to the north.

### 1. Preparation
Download the dataset :) 

### 2. Set-Up
This code was tested using Python 3.12 on Ubuntu 24.04. To start, clone the repository and then use the provided requirements.txt to install the dependencies:

`pip install -r requirements.txt`

### 3. Pre-processing step: SLAM evaluation data to pickle
As the D-GNSS setup did not measure orientation of the camera, we approximate it using the results from the evaluated visual-inertial SLAM/Odometry algorithms. The idea is that, after aligning trajectories and camera poses to the D-GNSS ground truth, transformed into metric coordinates for an arbitrary ENU frame, the pose estimation should be accurate enough to provide a very good guess of the real orientatio of the cameras. Therefore, we can use this information to better discern between positive and negative samples for place recognition. 

The script `slam_poses_to_pandas.py` reads the output `.txt` files from the `Eval` folder, and transforms them into a pandas dataframe. The dataframe will contain SLAM/VO results in the following tabular form: 

| Timestamp       | Rotation                       | Position  |
| --------        | -------                        |-------    |
| posix_time [s]  | Quaternion **(w, x, y, z)**    | [x, y, z] |

Usage:
1) From the root package folder: ```python3 scripts/slam_poses_to_pandas.py ${path_to_config}``` (e.g., ```python3 scripts/slam_poses_to_pandas.py cfg/config.yaml```)
2) This will write pickle files in a folder names `processed` from the root dataset path

### 4. Sample Generation
This step foresees the generation of all data samples for each sequence. The script `create_dataset.py` reads the content of all bagfiles in the `Bagfile` folder, and associates data from the various topics, writing links and data into a pandas dataframe, then stored as a pickle to disk. 
In this step, the results of SLAM evaluations, pre-processed in Sec.2, are used to approximate an orientation (with respect to the North) for each image/scan pair, as the ground truth is only from D-GNSS without magnetometer or full RTK. To account for the yaw-drift, which is inevitable in the context of V-SLAM, camera trajectories are split in an user-defined manner (default is 3 splits) and independently aligned to the ground truth (x, y, z in ENU frame). The corresponding rotation is applied to the camera poses, and the northing is stored.

Usage: 
1) From the package root: ```python3 scripts/create_dataset.py ${path_to_config}``` (e.g. ```python3 scripts/create_dataset.py cfg/config.yaml```)

### 5. Lidar Histograms Generation
This script `create_lidar_histograms.py` processes LiDAR point cloud data stored in the pickle files and generates histograms representing the distribution of depth (Z-axis) values. The generated histograms will later be used in the visualization of the interactive plot.

Usage:
1) From the package root:```python3 scripts/create_lidar_histograms.py ${path_to_config}```
`
### 6. Visualization
The results can be qualitatively inspected using the provided script `make_maps_in_bokeh.py`, which overlays subsampled images (with LiDAR point projections) on geo-references positions in an interactive map. 

Usage: 
1) From the package root: ```python3 scripts/make_maps_in_bokeh.py ${path_to_config}```

This will look in the dataset folders for saved pickles, that contain a dataframe for each entire bagfile, and create an interactive Bokeh plot showing the global position and orientation of collected image/scan samples and pre-vieweing the image with lidar overlay when hovering with the mouse cursor. 

### 7. Interactive plot
Visual analysis interactive tool for camera overlap. 
The script `interactive_overlap_maps.py` calculates the overlap between different camera views using two possible methods:

- An angle-based approach (compute_overlap_v1)
- A polygon intersection approach (compute_overlap_v2)

In addition, it considers occlusion detection by analyzing LiDAR depth data to adjust camera field-of-view ranges.

Usage:
1) From the package root: ```python3 scripts/interactive_overlap_maps.py  ${path_to_config}``` (e.g. ```python3 scripts/interactive_overlap_maps.py cfg/config.yaml```)

Example Bokeh plot:
![bokeh_plot](doc/toolkit.gif)

If you find this project useful for your work, consider to cite 
```
@article{giubilato2026s3li,
  title={The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments},
  author={Giubilato, Riccardo and M{\"u}ller, Marcus Gerhard and Sewtz, Marco and Gonzalez, Laura Alejandra Encinar and Folkesson, John and Triebel, Rudolph},
  journal={2026 IEEE Aerospace Conference},
  year={2026}
}

@ARTICLE{9813579,
  author={Giubilato, Riccardo and Stürzl, Wolfgang and Wedler, Armin and Triebel, Rudolph},
  journal={IEEE Robotics and Automation Letters}, 
  title={Challenges of SLAM in Extremely Unstructured Environments: The DLR Planetary Stereo, Solid-State LiDAR, Inertial Dataset}, 
  year={2022},
  volume={7},
  number={4},
  pages={8721-8728},
  doi={10.1109/LRA.2022.3188118}
}
```