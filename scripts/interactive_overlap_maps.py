import yaml
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, TapTool, CustomJS, Div, Arrow, OpenHead
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
import os
import argparse
import progressbar
from shapely.geometry import Polygon
import sequences_definition

from create_dataset import build_cinfo, resize_camera_info_and_rect_maps, CameraInfo


"""
Generator to return dataframes for each sequence
"""
def read_and_merge_dataframes(sequences, base_path):
    df = None

    for sequence in sequences:
        path = os.path.join(base_path, sequence, sequence + '.pkl')

        if not os.path.isfile(path):
            print("Could not sequence from pickle " + str(path))
            continue

        df_current = pd.read_pickle(path)
        df_current['overlay'] = df_current['overlay'].replace("s3li_crater_inout", "s3li_zcrater_inout")

        if df is None:
            df = df_current
        else:
            df = pd.concat([df, df_current], ignore_index=True)

        print("Read sequence from pickle " + str(path))
    return df

"""
Triangulate intersection between two directions. pos0 and pos1 must be numpy arrays of lenght 2 (x, y). the angles
are positive "northing", i.e. positive counter-clockwise from the "y" direction.
Returns:
    - intersection point
    - signed distance from the first point
    - signed distance from the second point
"""
def triangulate_intersection(pos0, ang0, pos1, ang1):
    a = np.array([[-np.sin(ang0), np.sin(ang1)], [np.cos(ang0), -np.cos(ang1)]])
    b = - np.array([pos0[0] - pos1[0], pos0[1] - pos1[1]])
    t = np.linalg.solve(a, b)
    return pos0 + t[0] * np.array([np.sin(ang0), -np.cos(ang0)]), t[0], t[1]

"""
Returns:
    - lateral distance between (directed) pos0 to pos1
    - longitudinal distance between (directed) pos0 to pos1
"""
def lateral_longitudinal_distances(pos0, ang0, pos1):
    p01 = np.array([pos1[0] - pos0[0], pos1[1] - pos0[1]])
    d0 = np.array([-np.sin(ang0), np.cos(ang0)])
    a01 = np.arccos(np.dot(d0, p01) / np.linalg.norm(p01, 2))
    return (np.linalg.norm(p01 * np.cos(a01), 2),
            np.linalg.norm(p01 * np.sin(a01), 2))
"""
Angles are northings in anti-clockwise direction. 0° is north, 90° is west, etc..
"""

def compute_overlap_v1(idx0, idx1, pos0, ang0, pos1, ang1, hor_fov = 45.0):
    if pos0[0] > pos1[0]:
        pos0, pos1 = pos1, pos0
        ang0, ang1 = ang1, ang0

    ang0_positive = (180.0 * ang0 / np.pi) % 360
    ang1_positive = (180.0 * ang1 / np.pi) % 360
    ang_diff = ang0_positive - ang1_positive
    ang_difference =  180 - np.abs(np.abs(ang_diff) - 180)


    if pos0[0] < pos1[0] and ang_diff>0 and pos0[1]>pos1[1]-10  and ang_difference<90:
        angular_overlap_ratio=1
    elif pos0[0] < pos1[0] and ang_diff<0 and ang_difference<90:
        angular_overlap_ratio=1
    else:
        angular_overlap_ratio = max(hor_fov - abs(ang_difference), 0.0) / hor_fov

    lateral_distance, longitudinal_distance = lateral_longitudinal_distances(pos0, ang0, pos1)
    position_correction_lateral = 1.0 - 1.0 / (1.0 + np.exp(-lateral_distance + 5.0)) #15
    position_correction_forward = 1.0 - 1.0 / (1.0 + np.exp(-longitudinal_distance + 10.0)) #40
    return angular_overlap_ratio * np.min([position_correction_lateral, position_correction_forward]), ang_difference, lateral_distance, longitudinal_distance

def create_progressbar():
    widgets = [' [', progressbar.Percentage(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ')']
    return progressbar.ProgressBar(widgets=widgets, maxval=100)

"""
Intersection of camera frustrums as projectet to the ground
penalize_angle_threshold   ["threshold", "decay", "none"]
"""
def compute_overlap_v2(pos0, ang0, pos1, ang1, hor_fov=90.0,
                       fov_range1=25.0, fov_range2=25.0,
                       min_fov_range1=0.0, min_fov_range2=0.0,
                       penalize_large_angles_method="none",
                       angle_threshold=360.0, # deg
                       debug=False):
    """
    Computes the intersection area between the FOVs of two cameras.

    Parameters:
        pos0, pos1: (x, y) positions of the cameras
        ang0, ang1: angles in degrees (northing, anti-clockwise)
        hor_fov: horizontal field of view in degrees
        fov_range1: max range of the first camera's FOV
        fov_range2: max range of the second camera's FOV
        penalize_large_angles_method   ["threshold", "decay", "none"]
        angle_threshold: threshold for angle between camera headings
        debug toggle dump to disk

    Returns:
        overlap_ratio: ratio of intersection area to total FOV area
        ang_difference: absolute angle difference
        lateral_distance: distance perpendicular to cam0's view
        longitudinal_distance: distance along cam0's view direction
    """

    # Get the FOV triangle vertices for both cameras
    fov0 = get_fov_triangle(pos0, ang0, hor_fov, fov_range1, min_fov_range=min_fov_range1)
    fov1 = get_fov_triangle(pos1, ang1, hor_fov, fov_range2, min_fov_range=min_fov_range2)

    # Compute intersection area
    poly0 = Polygon(fov0)
    poly1 = Polygon(fov1)
    intersection = poly0.intersection(poly1)
    intersection_area = intersection.area if intersection.is_valid else 0.0

    # Compute the total area of a single FOV triangle (max of both cameras)
    fov_area = max(Polygon(fov0).area, Polygon(fov1).area)

    # Compute final overlap ratio
    overlap_ratio = 0.0
    if fov_area > 0:
        overlap_ratio = intersection_area / fov_area
    else:
        print("WEIRD! Null polygon area")
        print(f"areas: {Polygon(fov0).area, Polygon(fov1).area}")
        print(f"intersection_area: {intersection_area}")

    # Compute angular difference and positional distances
    ang_difference = abs(ang0 - ang1) % (2 * np.pi)
    ang_difference = np.pi - abs(ang_difference - np.pi)
    ang_difference = 180.0 * ang_difference / np.pi
    lateral_distance, longitudinal_distance = lateral_longitudinal_distances(pos0, ang0, pos1)

    # Angle-base overlap penalization
    if penalize_large_angles_method == "threshold":
        if ang_difference > angle_threshold:
            overlap_ratio = 0.0
    elif penalize_large_angles_method == "decay":
        print("Warning in compute_overlap_v2 -> decaying angle cost not yet implemented")
    elif penalize_large_angles_method == "none":
        pass
    else:
        import sys
        print(f"Error in compute_overlap_v2 -> penalize angle method unknown: {penalize_large_angles_method}")
        sys.exit(1)

    if debug:
        print(f"pos_0 {pos0}")
        print(f"pos_1 {pos1}")
        print(f"ang_0 {ang0}")
        print(f"ang_1 {ang1}")
        print(f"intersection_area: {intersection_area}")
        print(f"overlap_ratio: {overlap_ratio}")
        print(f"Polygon 0 {Polygon(fov0)}")
        print(f"Polygon 1 {Polygon(fov1)}")
        print(f"area_0: {Polygon(fov0).area}")
        print(f"area_1: {Polygon(fov1).area}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for poly, color in zip([Polygon(fov0), Polygon(fov1)], ['skyblue', 'salmon']):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec='black', label=f'Area: {poly.area}')

        ax.set_aspect('equal')
        ax.legend()
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.savefig('/tmp/polygons.png')

    return overlap_ratio, ang_difference, lateral_distance, longitudinal_distance


def get_fov_triangle(pos, angle, hor_fov, fov_range, min_fov_range = 0.0):
    """
    Computes the vertices of the camera's FOV frustrum.

    Parameters:
        pos: (x, y) position of the camera
        angle: viewing angle in radians
        hor_fov: horizontal field of view in degrees
        fov_range: max measurable depth from an associated range sensor
        min_fov_range: min measurable depth from an associated range sensor

    Returns:
        List of 4 (x, y) points defining the FOV trapezoid, with max range and min range as parsed
    """
    half_fov = np.radians(hor_fov/2)
    angle = angle - 0.5 * np.pi

    # Compute left and right FOV boundary angles
    #left_angle = angle + half_fov
    #right_angle = angle - half_fov

    # Compute frustrum points, pointing to positive y (north)
    left_vertex = (-fov_range * np.tan(half_fov),
                    fov_range)

    right_vertex = (fov_range * np.tan(half_fov),
                    fov_range)

    left_mid_vertex = (-min_fov_range * np.tan(half_fov),
                        min_fov_range)

    right_mid_vertex = (min_fov_range * np.tan(half_fov),
                        min_fov_range)

    # Rotate points and sum to origin
    # var x_rot = x * Math.cos(angle) - y * Math.sin(angle);
    # var y_rot = x * Math.sin(angle) + y * Math.cos(angle);
    points_out = []
    for p in [left_mid_vertex, left_vertex, right_vertex, right_mid_vertex]:
        points_out.append(
            (pos[0] + p[0] * np.cos(angle) - p[1] * np.sin(angle),
             pos[1] + p[0] * np.sin(angle) + p[1] * np.cos(angle))
        )

    return points_out


def estimate_camera_depth(lidar_depths, percentile):
    """
    Estimate the adjusted field of view for a camera based on LiDAR depth histogram.
    Args:
        lidar_depths (numpy array): Array of depth values from LiDAR.
        percentile (float): value from the depth distribution

    Returns:
        float or None: Estimated max visible range for the camera, or None if no occlusion is detected.
    """
    # TODO: detection of occlusions
    return np.percentile(lidar_depths, percentile)


def create_confusion_matrix_for_sample_overlap(df, use_overlap_v2=False,
        override_lidar_range = False,
        min_range_fov = 3.0,
        max_range_fov = 20.0,
        horizontal_fov = 90.0,
        penalize_large_angles_method="none",
        cross_view_angle_threshold=360.0,
        min_timeout_between_samples=30.0,
        min_overlap_for_positive_match=0.5):
    confusion_matrix = np.zeros((df.shape[0], df.shape[0]))
    angular_difference = np.zeros((df.shape[0], df.shape[0]))
    lateral_position_difference = np.zeros((df.shape[0], df.shape[0]))
    longitudinal_position_difference = np.zeros((df.shape[0], df.shape[0]))
    fov_intersection_mercator = np.zeros((df.shape[0], df.shape[0], 2))

    # Compute max visible depth for each sample beforehand
    # We evaluate the distribution of depths from the LiDAR (already preprocessed by thresholding min intensity
    # in the create_dataset function), and get the 0 percentile for the min and and the 80-th percentile
    # NOTE: optionally, these are overridden later
    max_visible_depths = np.array([
        estimate_camera_depth(df.iloc[idx]['point_cloud'][:, 2], 90) for idx in range(df.shape[0])
    ])
    min_visible_depths = np.array([
        estimate_camera_depth(df.iloc[idx]['point_cloud'][:, 2], 0) for idx in range(df.shape[0])
    ])

    # Print occlusion information as clickable links
    #print("Indices with occlusion and corresponding histogram & overlay files:")
    #for idx, max_depth in enumerate(max_visible_depths):
    #    if max_depth is not None:  # This means occlusion was detected
    #        hist_path = df.iloc[idx]['histogram_path']
    #        overlay_path = df.iloc[idx]['overlay']
    #        hist_url = f"file://{hist_path}"  # Convert to clickable file link
    #        overlay_url = f"file://{overlay_path}"#

    #        print(f"Index: {idx}")
    #        print(f"  Histogram: \033]8;;{hist_url}\033\\{hist_path}\033]8;;\033\\")
    #        print(f"  Overlay: \033]8;;{overlay_url}\033\\{overlay_path}\033]8;;\033\\\n")

    # Set a "timeout" between consecutive views
    timestamps = df['time_stamp'].to_list()

    # Initialize list of tuples (path_to_left, path_to_right, overlap) to dump to
    # disk as picke file
    out_samples = []

    progress = create_progressbar()
    progress.start()

    for idx_0, sample_0 in enumerate(df.itertuples()):

        progress.update((float(idx_0) / len(df)) * 100)
        
        for idx_1, sample_1 in enumerate(df.itertuples()):

            if np.abs(timestamps[idx_0] - timestamps[idx_1]) < min_timeout_between_samples:
               continue

            #overlap function option 2
            if use_overlap_v2:
                # Determine max visible depth for sample_0
                max_visible_depth_1 = max_visible_depths[idx_0]
                max_visible_depth_2 = max_visible_depths[idx_1]
                min_visible_depth_1 = min_visible_depths[idx_0]
                min_visible_depth_2 = min_visible_depths[idx_1]

                # Override lidar ranges if requested
                if override_lidar_range:
                    max_visible_depth_1 = max_visible_depth_2 = max_range_fov
                    min_visible_depth_1 = min_visible_depth_2 = min_range_fov

                # Occlusion detected, use adjusted range
                overlap_score, ang_diff, lateral_diff, longitudinal_diff = compute_overlap_v2(
                    np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                    np.array([sample_1.x, sample_1.y]), sample_1.orientation,
                    fov_range1=max_visible_depth_1, fov_range2=max_visible_depth_2,
                    hor_fov=horizontal_fov,
                    min_fov_range1=min_visible_depth_1, min_fov_range2=min_visible_depth_2,
                    penalize_large_angles_method=penalize_large_angles_method,
                    angle_threshold=cross_view_angle_threshold
                )
            else:
                overlap_score, ang_diff, lateral_diff, longitudinal_diff = (
                    compute_overlap_v1(idx_0, idx_1, np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                                    np.array([sample_1.x, sample_1.y]), sample_1.orientation, hor_fov=horizontal_fov))

            fov_intersection, _, _ = triangulate_intersection(
                np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                np.array([sample_1.x, sample_1.y]), sample_1.orientation)
            fov_intersection_mercator[idx_0, idx_1] = fov_intersection
            confusion_matrix[idx_0, idx_1] = overlap_score
            angular_difference[idx_0, idx_1] = ang_diff
            lateral_position_difference[idx_0, idx_1] = lateral_diff
            longitudinal_position_difference[idx_0, idx_1] = longitudinal_diff

            if overlap_score > min_overlap_for_positive_match:
                out_samples.append((sample_0.img_path, sample_1.img_path, overlap_score))

    return confusion_matrix, angular_difference, lateral_position_difference, longitudinal_position_difference, fov_intersection_mercator, min_visible_depths,  max_visible_depths, out_samples

def plot_confusion_matrix(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    plt.savefig('/tmp/confusion_matrix.png')

def wgs84_to_web_mercator(df, lon, lat):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df

def gen_triangle_polygon_from_fov(fov=45.0, min_range=3.0, max_range=20.0):
    dx = max_range * np.tan(.5 * np.pi / 180.0 * fov)
    return [[dx * min_range/max_range, min_range], [dx, max_range], [-dx, max_range], [-dx * min_range/max_range, min_range]]

def compute_map_ranges(df, scale=1.0):
    x = df['x'].values
    y = df['y'].values

    # The range for the map extents is derived from the lat/lon fields. This way the map is automatically centered on the plot elements.
    x_min = int(x.mean() - (scale * 50))
    x_max = int(x.mean() + (scale * 50))
    y_min = int(y.mean() - (scale * 50))
    y_max = int(y.mean() + (scale * 50))

    return x_min, x_max, y_min, y_max

def create_bokeh_interactive_plot(dataframe, confusion_matrix, angular_difference,
                                lateral_pos_diff, longitudinal_pos_diff, fov_intersections,
                                min_fov_ranges, max_fov_ranges, horizontal_fov=90.0):
    x_range, y_range = confusion_matrix.shape[0], confusion_matrix.shape[1]  # Grid edges

    title_string = "Ground-truth overlap of pairs (Click for infos!)"
    p = figure(width=600, height=600, x_range=(0, x_range), y_range=(0, y_range), active_scroll ="wheel_zoom",title=title_string)
    p.image(image=[confusion_matrix], x=0, y=0, dw=x_range, dh=y_range, palette="Viridis256")

    # Create a scatter plot overlay to detect clicks
    x_grid, y_grid = np.meshgrid(np.arange(x_range), np.arange(y_range))
    n_elements = len(dataframe['overlay'].tolist())

    dataframe["arrow_x"] = dataframe['x'] + np.cos(dataframe["orientation"].values) * 5.0
    dataframe["arrow_y"] = dataframe['y'] + np.sin(dataframe["orientation"].values) * 5.0
    source = ColumnDataSource(data=dict(x=x_grid.ravel().tolist(),
                                        y=y_grid.ravel().tolist(),
                                        value=confusion_matrix.flatten().tolist(),
                                        angular_difference=angular_difference.flatten().tolist(),
                                        lateral_position_difference=lateral_pos_diff.flatten().tolist(),
                                        longitudinal_position_difference=longitudinal_pos_diff.flatten().tolist(),
                                        x_mercator=n_elements*dataframe['x'].tolist(),
                                        y_mercator=n_elements*dataframe['y'].tolist(),
                                        orientation=n_elements*dataframe['orientation'].tolist(),
                                        latitude=n_elements*dataframe['latitude'].tolist(),
                                        longitude=n_elements*dataframe['longitude'].tolist(),
                                        arrow_x=n_elements*dataframe['arrow_x'].tolist(),
                                        arrow_y=n_elements*dataframe['arrow_y'].tolist(),
                                        fov_int_x=fov_intersections[:, :, 0].flatten().tolist(),
                                        fov_int_y=fov_intersections[:, :, 1].flatten().tolist(),
                                        min_ranges=min_fov_ranges.tolist(),
                                        max_ranges=max_fov_ranges.tolist(),
                                        image_overlay_path=n_elements*dataframe['overlay'].tolist(),
                                        image_path=n_elements*dataframe['img_path'].tolist(),
                                        lidar_hist_path=n_elements*dataframe['histogram_path'].tolist(),
                                        sequence_name=n_elements*dataframe['seq_name'].tolist()))

    # JavaScript callback for popups
    image_1_div = Div(text="""""")
    image_2_div = Div(text="""""")
    top_info_div = Div(text="""""")
    source_scatter = ColumnDataSource(data=dict(x=(x_grid+0.5).ravel().tolist(),
                                                y=(y_grid+0.5).ravel().tolist()))
    p.scatter('x', 'y', source=source_scatter, size=10, alpha=0, nonselection_alpha=0)

    # Just for highlighting the click
    source_click = ColumnDataSource(data=dict(x=[], y=[], size=[], color=[]))
    p.scatter(x='x', y='y', marker='o+', size=20, line_color='red', fill_alpha=0,
            source=source_click, name='click_highlight')
    # Map view to highlight clicked point
    x_min, x_max, y_min, y_max = compute_map_ranges(dataframe)
    plot_map = figure(
        width=400, height=400,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        active_scroll ="wheel_zoom",
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_type='mercator',
        y_axis_type='mercator',
        title='Top-down view of sample pair \n Frustrums are built using LiDAR ranges'
    )

    # Show pairs of positions and arrows in the top-down view
    source_pinpoints = ColumnDataSource(data=dict(x=[0, 0],
                                                y=[0, 0],
                                                arrow_x=[0, 0],
                                                arrow_y=[0, 0],
                                                fov_int_x=[0, 0],
                                                fov_int_y=[0, 0]))
    plot_map.scatter('x', 'y', source=source_pinpoints)
    plot_map.add_layout(Arrow(end=OpenHead(line_color="firebrick", line_width=4.0, size=10.0),
                            x_start='x', y_start='y', x_end='arrow_x', y_end='arrow_y', source=source_pinpoints))
    plot_map.add_tile(tile_source="esri_imagery")

    # Show FOVs as triangles
    source_fovs_query = ColumnDataSource(data=dict(x=[0, 0, 0], y=[0, 0, 0]))
    source_fovs_train = ColumnDataSource(data=dict(x=[0, 0, 0], y=[0, 0, 0]))
    plot_map.patch('x', 'y', source=source_fovs_query, color="firebrick", alpha=0.3)
    plot_map.patch('x', 'y', source=source_fovs_train, color="deepskyblue", alpha=0.3)

    # Callback on clicks over confusion matrix
    callback = CustomJS(args=dict(source=source, source_click=source_click, source_scatter=source_scatter,
                                source_pinpoints=source_pinpoints,
                                source_fovs_query=source_fovs_query, source_fovs_train=source_fovs_train,
                                horizontal_fov=horizontal_fov,
                                x_range=plot_map.x_range, y_range=plot_map.y_range,
                                info_div=top_info_div, image_1_div=image_1_div, image_2_div=image_2_div), code="""
        var selected = source_scatter.selected.indices[0];
        if (selected !== undefined) {
            var value = source.data['value'][selected];
            var ang_diff = source.data['angular_difference'][selected];
            var lat_pos_diff = source.data['lateral_position_difference'][selected];
            var long_pos_diff = source.data['longitudinal_position_difference'][selected];
            var latitude = source.data['latitude'][selected];
            var longitude = source.data['longitude'][selected];

            var ang_diff_text = "Angular diff: " + ang_diff.toFixed(2) + " °";
            var lat_pos_diff_text = "Position diff (lateral): " + lat_pos_diff.toFixed(2) + " [m]";
            var long_pos_diff_text = "Position diff (longitudinal): " + long_pos_diff.toFixed(2) + " [m]";
            var score_text = "Overlap score: " + value.toFixed(2);

            var n_images = Math.sqrt(source.data['x'].length)
            var id_query = source.data['x'][selected];
            var id_train = source.data['y'][selected];

            var img_element_1_text = "Id: " + id_query;
            var img_element_2_text = "Id: " + id_train;
            var img_path_1 = source.data['image_path'][id_query];
            var img_overlay_path_1 = source.data['image_overlay_path'][id_query];
            var lidar_hist_path_1 = source.data['lidar_hist_path'][id_query];

            var img_path_2 = source.data['image_path'][id_train];
            var img_overlay_path_2 = source.data['image_overlay_path'][id_train];
            var lidar_hist_path_2 = source.data['lidar_hist_path'][id_train];

            info_div.text = `<div style="
                        gap: 20px;
                        background: rgba(0, 0, 0, 0.1);  /* Semi-transparent dark background */
                        padding: 15px;
                        border-radius: 10px;  /* Rounded corners */
                        border: 2px solid white;  /* White solid border */
                        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);  /* Soft shadow */
                        max-width: 450px;
                        margin: auto;">
                    <span id="image-info-2" style="display:block;"> ${ang_diff_text} </span>
                    <span id="image-info-2" style="display:block;"> ${lat_pos_diff_text} </span>
                    <span id="image-info-2" style="display:block;"> ${long_pos_diff_text} </span>
                    <span id="image-info-2" style="display:block;"> ${score_text} </span></div>`

            image_1_div.text = `<div style="
                        gap: 20px;
                        background: rgba(0, 0, 0, 0.1);  /* Semi-transparent dark background */
                        padding: 15px;
                        border-radius: 10px;  /* Rounded corners */
                        border: 2px solid white;  /* White solid border */
                        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);  /* Soft shadow */
                        max-width: 450px;
                        margin: auto;">
                    <span id="image-info-1" style="display:block; color: #910707;"> ${img_element_1_text} </span>
                    <span id="image-info-1" style="display:block; color: #910707;"> ${source.data['sequence_name'][id_query]} </span>
                    <img id="popup-image-1" src="${img_path_1}" style="width:300px; display:block;"><br>
                    <img id="popup-image-1" src="${img_overlay_path_1}" style="width:300px; display:block;"><br>
                    <img id="popup-image-1" src="${lidar_hist_path_1}" style="width:300px; display:block;"></div>`;

            image_2_div.text = `<div style="
                        gap: 20px;
                        background: rgba(0, 0, 0, 0.1);  /* Semi-transparent dark background */
                        padding: 15px;
                        border-radius: 10px;  /* Rounded corners */
                        border: 2px solid white;  /* White solid border */
                        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);  /* Soft shadow */
                        max-width: 450px;
                        margin: auto;">
                    <span id="image-info-2" style="display:block; color: #009dda;"> ${img_element_2_text} </span>
                    <span id="image-info-2" style="display:block; color: #009dda;"> ${source.data['sequence_name'][id_train]} </span>
                    <img id="popup-image-2" src="${img_path_2}" style="width:300px; display:block;"><br>
                    <img id="popup-image-2" src="${img_overlay_path_2}" style="width:300px; display:block;"><br>
                    <img id="popup-image-2" src="${lidar_hist_path_2}" style="width:300px; display:block;"></div>`;

            source_click.data['x'] = []
            source_click.data['y'] = []
            source_click.data['color'] = []
            source_click.data['size'] = []
            source_click.data['x'].push(id_query + 0.5);
            source_click.data['y'].push(id_train + 0.5);
            source_click.data['color'].push('red');
            source_click.data['size'].push(10);
            source_click.change.emit();

            source_pinpoints.data['x'] = [source.data['x_mercator'][id_query], source.data['x_mercator'][id_train]];
            source_pinpoints.data['y'] = [source.data['y_mercator'][id_query], source.data['y_mercator'][id_train]];
            source_pinpoints.data['arrow_x'] = [source.data['arrow_x'][id_query], source.data['arrow_x'][id_train]];
            source_pinpoints.data['arrow_y'] = [source.data['arrow_y'][id_query], source.data['arrow_y'][id_train]];
            source_pinpoints.data['line_color'] = ['red', 'blue'];
            source_pinpoints.change.emit();

            var lat_rad = latitude * Math.PI / 180;
            var merc_scale = 1.0 / Math.cos(lat_rad); // The "stretch" factor

            function create_fov_and_rotate_coordinates(angle, origin, min_fov_range, max_fov_range) {
                // Create variable-depth frustrum
                const hor_fov = horizontal_fov;
                const half_fov = (hor_fov / 2) * (Math.PI / 180);

                // Define the 4 vertices in local space
                const coords = [
                    [-min_fov_range * Math.tan(half_fov), min_fov_range], // left_mid_vertex
                    [-max_fov_range * Math.tan(half_fov), max_fov_range],         // left_vertex
                    [max_fov_range * Math.tan(half_fov), max_fov_range],          // right_vertex
                    [min_fov_range * Math.tan(half_fov), min_fov_range]  // right_mid_vertex
                ];

                angle = angle - 0.5 * Math.PI;
                var rotated_coords = [];
                for (var i = 0; i < coords.length; i++) {
                    // Multiply by merc_scale to fix the 'offset' look on the map
                    var x = coords[i][0] * merc_scale;
                    var y = coords[i][1] * merc_scale;

                    var x_rot = x * Math.cos(angle) - y * Math.sin(angle);
                    var y_rot = x * Math.sin(angle) + y * Math.cos(angle);
                    rotated_coords.push({x: x_rot + origin[0], y: y_rot + origin[1]});
                }
                return rotated_coords;
            }

            // Get the current coordinates of the polygons
            var new_coords_query = create_fov_and_rotate_coordinates(
                                                      source.data['orientation'][id_query],
                                                     [source.data['x_mercator'][id_query], source.data['y_mercator'][id_query]],
                                                      source.data['min_ranges'][id_query], source.data['max_ranges'][id_query]);
            var new_coords_train = create_fov_and_rotate_coordinates(
                                                      source.data['orientation'][id_train],
                                                     [source.data['x_mercator'][id_train], source.data['y_mercator'][id_train]],
                                                      source.data['min_ranges'][id_train], source.data['max_ranges'][id_train]);

            // Update the ColumnDataSource data with rotated coordinates
            source_fovs_query.data = {x: new_coords_query.map(function(coord) { return coord.x; }),
                                      y: new_coords_query.map(function(coord) { return coord.y; })};
            source_fovs_train.data = {x: new_coords_train.map(function(coord) { return coord.x; }),
                                      y: new_coords_train.map(function(coord) { return coord.y; })};

            // Trigger a change in the data
            source_fovs_query.change.emit();
            source_fovs_train.change.emit();

            var x_center = 0.5 * (source_pinpoints.data['x'][0] + source_pinpoints.data['x'][1]);
            var y_center = 0.5 * (source_pinpoints.data['y'][0] + source_pinpoints.data['y'][1]);
            var half_size = 0.5 * Math.max(Math.abs(source_pinpoints.data['x'][0] - source_pinpoints.data['x'][1]) + 200,
                                        Math.abs(source_pinpoints.data['y'][0] - source_pinpoints.data['y'][1]) + 200);
            x_range.start = x_center - half_size;
            x_range.end = x_center + half_size;
            y_range.start = y_center - half_size;
            y_range.end = y_center + half_size;
            source_scatter.selected.indices = [];
        }
    """)

    # Attach TapTool
    tap_tool = TapTool(callback=callback)
    p.add_tools(tap_tool)
    show(row(p, plot_map, column(top_info_div, row(image_1_div, image_2_div))))

# Returns the horizontal field of view in degrees
def horizontal_fov_from_camera_info(camera_info: CameraInfo):
    return 2 * np.arctan(camera_info.width / (2 * camera_info.K[0, 0])) * 180.0 / np.pi

def write_ground_truth_to_disk(out_samples, base_path):
    import pickle
    from datetime import datetime
    now = datetime.now()
    filename = base_path + '/gt_samples' + now.strftime("%m-%d-%Y-%H-%M-%S") + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(out_samples, file)
    print(f"Saved gt-poses to: {filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a Bokeh plot, to inspect the generated dataset')
    parser.add_argument('params', type=str, help='path to config file')

    # Read parameters
    args = parser.parse_args()
    with open(args.params, 'rb') as f:
        params = yaml.safe_load(f.read())

    skip = params['interactive_overlap_maps']['skip']
    use_overlap_with_lidar_range_aid = params['interactive_overlap_maps']['overlap_comp_type'] == 2
    override_lidar_range = params['interactive_overlap_maps']['override_lidar_range']
    min_range_fov = params['interactive_overlap_maps']['min_fov_range']
    max_range_fov = params['interactive_overlap_maps']['max_fov_range']
    penalize_large_angles_method = params['interactive_overlap_maps']['penalize_large_angles_method']
    cross_view_angle_threshold = params['interactive_overlap_maps']['cross_view_angle_threshold']
    min_timeout_between_samples = params['interactive_overlap_maps']['min_timeout_between_samples']
    min_overlap_for_positive_match = params['interactive_overlap_maps']['min_overlap_for_positive_match']
    plot_in_browser = params['interactive_overlap_maps']['plot_in_browser']
    print(f"loaded params:\n{yaml.dump(params, default_flow_style=False)}")

    # Paths
    base_path = params['base_path'] + '/dataset'
    path_to_camera_yaml = params['base_path'] + '/' + params['camchain_relative_path']

    # For this version of the toolkit, we assume that the camera is always the same
    camera_info, _, _ = resize_camera_info_and_rect_maps(build_cinfo(path_to_camera_yaml),
        params['image_resize_factor'])
    horizontal_fov = horizontal_fov_from_camera_info(camera_info)
    print(f"Loaded camera with properties\nK: {camera_info.K} \nsize: ({camera_info.height}, {camera_info.width})\nhorizontal field of view: {horizontal_fov}")

    # Read all possible sequences, apply skip to samples to not overburden CPU for interactive viz
    sequences = sequences_definition.sequences
    df = read_and_merge_dataframes(sequences, base_path)
    df['sequence_index'] = pd.factorize(df['seq_name'])[0]
    df = df.iloc[::skip]
    df = wgs84_to_web_mercator(df, 'longitude', 'latitude')

    # Create confusion matrix
    res, ang_diff, lat_pos_diff, long_pos_diff, fov_intersections, min_ranges, max_ranges, out_samples = create_confusion_matrix_for_sample_overlap(
        df, use_overlap_with_lidar_range_aid, override_lidar_range, min_range_fov, max_range_fov,
        horizontal_fov=horizontal_fov,
        penalize_large_angles_method=penalize_large_angles_method,
        cross_view_angle_threshold=cross_view_angle_threshold,
        min_timeout_between_samples=min_timeout_between_samples,
        min_overlap_for_positive_match=min_overlap_for_positive_match)

    count_positive_samples = np.count_nonzero((res > 0.1))
    print("Number of overlapping samples: " + str(count_positive_samples) +
          " ({:.2%})".format(count_positive_samples / (0.5 * res.size)))
    print("Number of non overlapping samples: " + str(0.5 * res.size - count_positive_samples))

    plot_confusion_matrix(res)
    write_ground_truth_to_disk(out_samples, base_path)

    if plot_in_browser:
        create_bokeh_interactive_plot(df, res, ang_diff, lat_pos_diff, long_pos_diff,
            fov_intersections, min_ranges, max_ranges, horizontal_fov=horizontal_fov)

