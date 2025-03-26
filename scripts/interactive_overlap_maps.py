import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, TapTool, CustomJS, Div, Arrow, OpenHead
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
import os
import argparse

from shapely.geometry import Polygon

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
def compute_overlap_v1(pos0, ang0, pos1, ang1, hor_fov = 45.0):
    ang0_positive = (180.0 * ang0 / np.pi) % 360
    ang1_positive = (180.0 * ang1 / np.pi) % 360
    ang_difference =  180 - np.abs(np.abs(ang0_positive - ang1_positive) - 180)
    angular_overlap_ratio = max(hor_fov - abs(ang_difference), 0.0) / hor_fov
    lateral_distance, longitudinal_distance = lateral_longitudinal_distances(pos0, ang0, pos1)
    position_correction_lateral = 1.0 - 1.0 / (1.0 + np.exp(-lateral_distance + 15.0))
    position_correction_forward = 1.0 - 1.0 / (1.0 + np.exp(-longitudinal_distance + 40.0))
    return angular_overlap_ratio * np.min([position_correction_lateral, position_correction_forward]), ang_difference, lateral_distance, longitudinal_distance


def compute_overlap_v2(pos0, ang0, pos1, ang1, hor_fov=45.0, fov_range1=75.0, fov_range2=75.0):
    """
    Computes the intersection area between the FOVs of two cameras.

    Parameters:
        pos0, pos1: (x, y) positions of the cameras
        ang0, ang1: angles in degrees (northing, anti-clockwise)
        hor_fov: horizontal field of view in degrees
        fov_range: max range of the camera's FOV

    Returns:
        overlap_ratio: ratio of intersection area to total FOV area
        ang_difference: absolute angle difference
        lateral_distance: distance perpendicular to cam0's view
        longitudinal_distance: distance along cam0's view direction
    """

    # Get the FOV triangle vertices for both cameras
    fov0 = get_fov_triangle(pos0, ang0, hor_fov, fov_range1)
    fov1 = get_fov_triangle(pos1, ang1, hor_fov, fov_range2)

    # Compute intersection area
    poly0 = Polygon(fov0)
    poly1 = Polygon(fov1)
    intersection = poly0.intersection(poly1)
    intersection_area = intersection.area if intersection.is_valid else 0.0

    # Compute the total area of a single FOV triangle (max of both cameras)
    fov_area = max(Polygon(fov0).area, Polygon(fov0).area)

    # Compute final overlap ratio
    overlap_ratio = intersection_area / fov_area

    # Compute angular difference and positional distances
    ang_difference = abs(ang0 - ang1) % 360
    ang_difference = 180 - abs(ang_difference - 180)
    lateral_distance, longitudinal_distance = lateral_longitudinal_distances(pos0, ang0, pos1)

    return overlap_ratio, ang_difference, lateral_distance, longitudinal_distance


def get_fov_triangle(pos, angle, hor_fov, fov_range):
    """
    Computes the vertices of the camera's FOV triangle.

    Parameters:
        pos: (x, y) position of the camera
        angle: viewing angle in radians
        hor_fov: horizontal field of view in degrees
        fov_range: depth of the field of view

    Returns:
        List of three (x, y) points defining the FOV triangle
    """
    half_fov = np.radians(hor_fov/2)

    # Compute left and right FOV boundary angles
    left_angle = angle + half_fov
    right_angle = angle - half_fov

    # Compute triangle points
    left_vertex = (pos[0] - fov_range * np.sin(left_angle),
                   pos[1] - fov_range * np.cos(left_angle))

    right_vertex = (pos[0] - fov_range * np.sin(right_angle),
                    pos[1] - fov_range * np.cos(right_angle))

    return [pos, left_vertex, right_vertex]


def estimate_camera_fov(lidar_depths, peak_threshold=0.3):
    """
    Estimate the adjusted field of view for a camera based on LiDAR depth histogram.
    Args:
        lidar_depths (numpy array): Array of depth values from LiDAR.
        peak_threshold (float): Percentage threshold to define a significant peak.

    Returns:
        float or None: Estimated max visible range for the camera, or None if no occlusion is detected.
    """
    # Compute histogram
    hist, bin_edges = np.histogram(lidar_depths, bins=50, range=(0, np.max(lidar_depths)))

    # Normalize histogram counts
    hist_norm = hist / np.sum(hist)

    # Identify peak depth (first bin where the peak exceeds the threshold)
    for i, freq in enumerate(hist_norm):
        if freq > peak_threshold:
            estimated_fov = bin_edges[i]  # First depth where peak is found
            return estimated_fov  # Return occlusion depth

    return None  # No occlusion detected

def create_confusion_matrix_for_sample_overlap(df, use_overlap_v2=False):
    confusion_matrix = np.zeros((df.shape[0], df.shape[0]))
    angular_difference = np.zeros((df.shape[0], df.shape[0]))
    lateral_position_difference = np.zeros((df.shape[0], df.shape[0]))
    longitudinal_position_difference = np.zeros((df.shape[0], df.shape[0]))
    fov_intersection_mercator = np.zeros((df.shape[0], df.shape[0], 2))

    # Compute max visible depth for each sample beforehand
    max_visible_depths = np.array([
        estimate_camera_fov(df.iloc[idx]['point_cloud'][:, 2]) for idx in range(df.shape[0])
    ])

    # Print occlusion information as clickable links
    print("Indices with occlusion and corresponding histogram & overlay files:")
    for idx, max_depth in enumerate(max_visible_depths):
        if max_depth is not None:  # This means occlusion was detected
            hist_path = df.iloc[idx]['histogram_path']
            overlay_path = df.iloc[idx]['overlay']
            hist_url = f"file://{hist_path}"  # Convert to clickable file link
            overlay_url = f"file://{overlay_path}"

            print(f"Index: {idx}")
            print(f"  Histogram: \033]8;;{hist_url}\033\\{hist_path}\033]8;;\033\\")
            print(f"  Overlay: \033]8;;{overlay_url}\033\\{overlay_path}\033]8;;\033\\\n")


    for idx_0, sample_0 in enumerate(df.itertuples()):
        for idx_1, sample_1 in enumerate(df.itertuples()):
            if idx_0 <= idx_1:
                continue

            # From local "window"
            if sample_0.sequence_index == sample_1.sequence_index and abs(sample_0.time_stamp - sample_1.time_stamp) < 30.0:
                continue

            #overlap function option 2
            if use_overlap_v2:
                # Determine max visible depth for sample_0
                max_visible_depth_1 = max_visible_depths[idx_0]
                max_visible_depth_2 = max_visible_depths[idx_1]

                if max_visible_depth_1 is None and max_visible_depth_2 is None:
                    # No obstacle, use default range
                    overlap_score, ang_diff, lateral_diff, longitudinal_diff = compute_overlap_v2(
                        np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                        np.array([sample_1.x, sample_1.y]), sample_1.orientation
                    )
                elif max_visible_depth_2 is None:
                    # Occlusion detected, use adjusted range
                    overlap_score, ang_diff, lateral_diff, longitudinal_diff = compute_overlap_v2(
                        np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                        np.array([sample_1.x, sample_1.y]), sample_1.orientation,
                        fov_range1=max_visible_depth_1
                    )
                elif max_visible_depth_1 is None:
                    # Occlusion detected, use adjusted range
                    overlap_score, ang_diff, lateral_diff, longitudinal_diff = compute_overlap_v2(
                        np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                        np.array([sample_1.x, sample_1.y]), sample_1.orientation,
                        fov_range2=max_visible_depth_2
                    )
                else:
                    # Occlusion detected, use adjusted range
                    overlap_score, ang_diff, lateral_diff, longitudinal_diff = compute_overlap_v2(
                        np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                        np.array([sample_1.x, sample_1.y]), sample_1.orientation,
                        fov_range1=max_visible_depth_1, fov_range2=max_visible_depth_2
                    )
            else:
                overlap_score, ang_diff, lateral_diff, longitudinal_diff = (
                    compute_overlap_v1(np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                                    np.array([sample_1.x, sample_1.y]), sample_1.orientation))

            fov_intersection, _, _ = triangulate_intersection(
                np.array([sample_0.x, sample_0.y]), sample_0.orientation,
                np.array([sample_1.x, sample_1.y]), sample_1.orientation)
            fov_intersection_mercator[idx_0, idx_1] = fov_intersection
            confusion_matrix[idx_0, idx_1] = overlap_score
            angular_difference[idx_0, idx_1] = ang_diff
            lateral_position_difference[idx_0, idx_1] = lateral_diff
            longitudinal_position_difference[idx_0, idx_1] = longitudinal_diff
    return confusion_matrix, angular_difference, lateral_position_difference, longitudinal_position_difference, fov_intersection_mercator

def wgs84_to_web_mercator(df, lon, lat):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df

def gen_triangle_polygon_from_fov(fov=45.0, scale=1.0):
    dx = scale * np.tan(.5 * np.pi / 180.0 * fov)
    return [[0, 0], [dx, scale], [-dx, scale]]

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
                                  lateral_pos_diff, longitudinal_pos_diff, fov_intersections):
    x_range, y_range = confusion_matrix.shape[0], confusion_matrix.shape[1]  # Grid edges

    p = figure(width=600, height=600, x_range=(0, x_range), y_range=(0, y_range), active_scroll ="wheel_zoom",
        title='Ground-truth overlap of pairs (Click for infos!)')
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
    x_min, x_max, y_min, y_max = compute_map_ranges(dataframe, scale)
    plot_map = figure(
        width=400, height=400,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        active_scroll ="wheel_zoom",
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_type='mercator',
        y_axis_type='mercator',
        title='Top-down view of sample pair'
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
    plot_map.patch('x', 'y', source=source_fovs_train, color="firebrick", alpha=0.3)
    fov_triangle_coords = gen_triangle_polygon_from_fov(scale=75.0)

    # Callback on clicks over confusion matrix
    callback = CustomJS(args=dict(source=source, source_click=source_click, source_scatter=source_scatter,
                                  source_pinpoints=source_pinpoints,
                                  source_fovs_query=source_fovs_query, source_fovs_train=source_fovs_train,
                                  fov_triangle_coords=fov_triangle_coords,
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
            var img_path_2 = source.data['image_path'][n_images * id_query + id_train];
            var img_overlay_path_1 = source.data['image_overlay_path'][id_query];
            var img_overlay_path_2 = source.data['image_overlay_path'][n_images * id_query + id_train];
            var lidar_hist_path_1 = source.data['lidar_hist_path'][id_query];            
            var lidar_hist_path_2 = source.data['lidar_hist_path'][n_images * id_query + id_train];
            
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
                    <span id="image-info-1" style="display:block;"> ${img_element_1_text} </span> 
                    <span id="image-info-1" style="display:block;"> ${source.data['sequence_name'][id_query]} </span> 
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
                    <span id="image-info-2" style="display:block;"> ${img_element_2_text} </span> 
                    <span id="image-info-2" style="display:block;"> ${source.data['sequence_name'][id_train]} </span> 
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
            
            function rotate_coordinates(coords, angle, origin) {
                angle = angle - 0.5 * Math.PI
                var rotated_coords = [];
                for (var i = 0; i < coords.length; i++) {
                    var x = coords[i][0];
                    var y = coords[i][1];
                    var x_rot = x * Math.cos(angle) - y * Math.sin(angle);
                    var y_rot = x * Math.sin(angle) + y * Math.cos(angle);
                    rotated_coords.push({x: x_rot + origin[0], 
                                         y: y_rot + origin[1]});
                }
                return rotated_coords;
            }
        
            // Get the current coordinates of the polygons
            var new_coords_query = rotate_coordinates(fov_triangle_coords, 
                                                      source.data['orientation'][id_query],
                                                      [source.data['x_mercator'][id_query], source.data['y_mercator'][id_query]]);
            var new_coords_train = rotate_coordinates(fov_triangle_coords, 
                                                      source.data['orientation'][id_train],
                                                      [source.data['x_mercator'][id_train], source.data['y_mercator'][id_train]]);
        
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
        }
    """)


    # Attach TapTool
    tap_tool = TapTool(callback=callback)
    p.add_tools(tap_tool)
    show(row(p, plot_map, column(top_info_div, row(image_1_div, image_2_div))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a Bokeh plot, to inspect the generated dataset')
    parser.add_argument('base_path', type=str, help='path to root folder, which contains each individual'
                                                    'sequence as subfolder (generated by create_dataset.py)')
    parser.add_argument('skip', type=int, default=10, help='Factor to downsample the data for visualization')

    parser.add_argument('overlap_version', type=int, choices=[1, 2], help='Choose overlap computation version (1 or 2)')

    args = parser.parse_args()

    skip = args.skip
    scale = 1
    use_overlap_v2 = args.overlap_version == 2  # Set flag for overlap version

    # All possible sequences
    sequences = ["s3li_traverse_1",
                 "s3li_loops",
                 "s3li_traverse_2",
                 "s3li_crater",
                 "s3li_crater_inout",
                 "s3li_mapping",
                 "s3li_landmarks"]

    df = read_and_merge_dataframes(sequences, args.base_path)
    df['sequence_index'] = pd.factorize(df['seq_name'])[0]

    df = df.iloc[::skip]

    df = wgs84_to_web_mercator(df, 'longitude', 'latitude')

    res, ang_diff, lat_pos_diff, long_pos_diff, fov_intersections = create_confusion_matrix_for_sample_overlap(df, use_overlap_v2)

    count_positive_samples = np.count_nonzero((res > 0.1))
    print("Number of overlapping samples: " + str(count_positive_samples) +
          " ({:.2%})".format(count_positive_samples / (0.5 * res.size)))
    print("Number of non overlapping samples: " + str(0.5 * res.size - count_positive_samples))

    create_bokeh_interactive_plot(df, res, ang_diff, lat_pos_diff, long_pos_diff, fov_intersections)

