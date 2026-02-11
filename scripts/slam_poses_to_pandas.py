import pandas as pd
import argparse
import pickle
import numpy
import yaml
import os
from sequences_definition import sequences

pd.set_option("display.float_format", '{:.3f}'.format)
pd.set_option("display.width", 0)

from sympy.strategies.core import switch

# Careful! Quaternion is inverted. Check that with the actual result text!
def read_results_tum_format(path):
    with open(path, 'r') as file:
        data = numpy.loadtxt(file)
        df = pd.DataFrame(columns=["timestamp", "rotation", "position"])
        df["timestamp"] = data[:, 0]
        q = data[:, 4:8]
        df["rotation"] = numpy.column_stack((
            numpy.array(q[:, 3]),
            numpy.array(q[:, 0]),
            numpy.array(q[:, 1]),
            numpy.array(q[:, 2]))).tolist()
        df["position"] = data[:, 1:4].tolist()
        print(df)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transform .txt of S3LI SLAM results to Panda Dataframes')
    parser.add_argument('config', type=str, help="path to config file")
    args = parser.parse_args()
    with open(args.config, 'rb') as f:
         params = yaml.safe_load(f.read())
    algorithm = "rvio2"
    path = params["base_path"]

    if "ORB" in algorithm:
        pass

    if "BASALT" in algorithm or "rvio2" in algorithm:
        for sequence in sequences:
            for root, dirs, files in os.walk(os.path.join(path, 'eval', algorithm, sequence)):
                for file in files:
                    if file.endswith(".txt"):
                        print("Reading from {}".format(file))
                        df = read_results_tum_format(os.path.join(path, 'eval', algorithm, sequence, file))

                        if not os.path.isdir(os.path.join(path, "processed")):
                            os.mkdir(os.path.join(path, "processed"))

                        with open(os.path.join(path, "processed", sequence + '_poses.pkl'), 'wb') as handle:
                            print(handle)
                            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif "OPEN_VINS" in algorithm:
        pass
    elif "VINS" in algorithm:
        pass

