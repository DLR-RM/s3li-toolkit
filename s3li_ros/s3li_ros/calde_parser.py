def parse_matrix(text):
    """Parses a matrix string like '[1 0 0; 0 1 0; 0 0 1]' into a list of floats."""
    rows = text.strip('[]').split(';')
    return [float(x) for row in rows for x in row.strip().split()]


def parse_vector(text):
    """Parses a vector string like '[1 2 3]' into a list of floats."""
    return [float(x) for x in text.strip('[]').split()]


def parse_calibration_file(filename):
    """Parses the stereo calibration .txt file."""
    cameras = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    current_cam = None
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        if line.startswith('camera.'):
            key, value = line.split('=')
            parts = key.split('.')
            if len(parts) < 3:
                continue
            cam_id = int(parts[1])
            param = parts[2]
            if cam_id not in cameras:
                cameras[cam_id] = {}
            cameras[cam_id][param] = value.strip()
    return cameras