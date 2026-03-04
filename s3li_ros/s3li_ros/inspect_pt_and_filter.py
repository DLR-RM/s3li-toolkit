#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class IntensityFilter(Node):
    def __init__(self, input_topic, threshold):
        super().__init__('intensity_filter_node')
        self.threshold = float(threshold)

        # Publisher for the filtered cloud
        self.publisher = self.create_publisher(PointCloud2, input_topic+'_filtered', 10)

        # Subscriber to the raw cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.listener_callback,
            10)

        self.first_messaged_received = False

        self.get_logger().info(f"Filtering {input_topic} for intensity > {self.threshold}")

    def print_cloud_fields(self, msg):
        print(f"\n{'='*40}")
        print(f"POINTCLOUD2 METADATA")
        print(f"{'='*40}")
        print(f"Height: {msg.height}")
        print(f"Width:  {msg.width}")
        print(f"Point Step: {msg.point_step} bytes")

        print(f"\nFIELDS:")
        # msg.fields is a list of PointField objects
        for field in msg.fields:
            print(f" - Name: {field.name:<10} | Offset: {field.offset:<3} | Datatype: {field.datatype:<2} | Count: {field.count}")

        print(f"{'='*40}\n")

    def listener_callback(self, msg):
        if not self.first_messaged_received:
            self.print_cloud_fields(msg)
            self.first_messaged_received = True

        cloud_array = pc2.read_points(msg, skip_nans=True)
        mask = cloud_array['intensity'] > self.threshold
        filtered_array = cloud_array[mask]
        filtered_msg = pc2.create_cloud(msg.header, msg.fields, filtered_array)

        self.publisher.publish(filtered_msg)
        self.get_logger().info(f"Published {len(filtered_array)} points. (Intensity thresh: {self.threshold})", throttle_duration_sec=1.0)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 script.py /topic_name threshold_value")
        return

    topic = sys.argv[1]
    thresh = sys.argv[2]

    rclpy.init()
    node = IntensityFilter(topic, thresh)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
