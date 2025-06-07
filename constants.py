import os
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import map_pb2

CAM_NAME_INDEX_LABELS    = {
    dataset_pb2.CameraName.FRONT: "FRONT_CAM",
    dataset_pb2.CameraName.FRONT_LEFT: "FRONT_LEFT_CAM",
    dataset_pb2.CameraName.FRONT_RIGHT: "FRONT_RIGHT_CAM",
    dataset_pb2.CameraName.SIDE_LEFT: "SIDE_LEFT",
    dataset_pb2.CameraName.SIDE_RIGHT: "SIDE_RIGHT",
}
LASER_NAME_INDEX_LABELS  = {
    dataset_pb2.LaserName.TOP: "TOP_LIDAR",
    dataset_pb2.LaserName.FRONT: "FRONT_LIDAR",
    dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT_LIDAR",
    dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT_LIDAR",
    dataset_pb2.LaserName.REAR: "REAR_LIDAR",
}

MAP_ELEMENT_LABEL_INDEXES = {
    "lane": 0, 
    "road_line": 1, 
    "road_edge": 2, 
    "stop_sign": 3, 
    "crosswalk": 4, 
    "speed_bump": 5, 
    "driveway": 6
}

MAP_ELEMENT_INDEX_LABELS = {v : k for k, v in MAP_ELEMENT_LABEL_INDEXES.items()}

ROAD_LINE_TYPES = {
    map_pb2.RoadLine.TYPE_UNKNOWN: (
      (0, 0, 0), "solid"
    ),
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: (
      (190, 190, 190), "dash"
    ),
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: (
      (190, 190, 190), "solid"
    ),
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: (
        (190, 190, 190), "solid"
    ),
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: (
        (208, 255, 20), "dash"
    ),
    map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: (
        (208, 255, 20), "dash"
    ),
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: (
        (208, 255, 20), "solid"
    ),
    map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: (
        (208, 255, 20), "dash"
    ),
}
ROAD_EDGE_TYPES = {
    map_pb2.RoadEdge.TYPE_UNKNOWN: ((0, 0, 0), "solid"),
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: ((0, 255, 0), "solid"),
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: ((0, 255, 0), "solid"),
}
LANE_TYPES      = {
    map_pb2.LaneCenter.TYPE_UNDEFINED: ((0, 0, 0), "solid"),
    map_pb2.LaneCenter.TYPE_FREEWAY: ((255, 255, 255), "solid"),
    map_pb2.LaneCenter.TYPE_SURFACE_STREET: ((65, 105, 225), "solid"),
    map_pb2.LaneCenter.TYPE_BIKE_LANE: ((204, 0, 204), "solid"),
}

CROSSWALK_TYPE  = ((255, 165, 0), "solid")
SPEED_BUMP_TYPE = ((0, 128, 128), "solid")
STOP_SIGN_TYPE  = ((190, 20, 20), "solid")
DRIVEWAY_TYPE   = ((70, 150, 90), "solid")
DEFAULT         = ((0, 0, 0), "solid")

GCSFS_PERCEPTION_DATA    = "waymo_open_dataset_v_1_4_3/individual_files"
TEMP_DATA_PATH           = "data/temp"
DATA_PATH                = "data/waymo"
CAMERA_DATA_LOCAL_PATH   = os.path.join(DATA_PATH, "camera/data")
CAMERA_LABELS_LOCAL_PATH = os.path.join(DATA_PATH, "camera/labels")
POINT_CLOUD_LOCAL_PATH   = os.path.join(DATA_PATH, "lidar/point_clouds")
CAMERA_PROJ_LOCAL_PATH   = os.path.join(DATA_PATH, "lidar/camera_projections")
LASER_LABELS_LOCAL_PATH  = os.path.join(DATA_PATH, "lidar/labels")
MAP_MASKS_LOCAL_PATH     = os.path.join(DATA_PATH, "map_features/masks")
MAP_POLYLINES_LOCAL_PATH = os.path.join(DATA_PATH, "map_features/polylines")