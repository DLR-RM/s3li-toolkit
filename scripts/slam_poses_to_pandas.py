import pandas as pd
import argparse
import pickle
import numpy
import os

pd.set_option("display.float_format", '{:.3f}'.format)
pd.set_option("display.width", 0)

from sympy.strategies.core import switch

sequences = ["s3li_traverse_1",
             "s3li_loops",
             "s3li_traverse_2",
             "s3li_crater",
             "s3li_crater_inout",
             "s3li_mapping",
             "s3li_landmarks"]

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
    parser.add_argument('path', type=str, help='path to results (Eval) folder')
    parser.add_argument("algorithm", type=str, help="SLAM algorithm for the camera pose."
                                                    "Must match a subfolder of path")
    args = parser.parse_args()

    if "ORB" in args.algorithm:
        pass

    if "BASALT" in args.algorithm:
        for sequence in sequences:
            for root, dirs, files in os.walk(os.path.join(args.path, 'Eval', args.algorithm, sequence)):
                for file in files:
                    if file.endswith(".txt"):
                        print("Reading from {}".format(file))
                        df = read_results_tum_format(os.path.join(args.path, 'Eval', args.algorithm, sequence, file))

                        if not os.path.isdir(os.path.join(args.path, "processed")):
                            os.mkdir(os.path.join(args.path, "processed"))

                        with open(os.path.join(args.path, "processed", sequence + '_poses.pkl'), 'wb') as handle:
                            print(handle)
                            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif "OPEN_VINS" in args.algorithm:
        pass
    elif "VINS" in args.algorithm:
        pass

