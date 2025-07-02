import os
import gcsfs
import cv2
import zlib
import logging
import asyncio
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils.frame_utils import convert_range_image_to_point_cloud
from utils.io_utils import delete_path, save_pickle_file, save_json_file
from typing import Dict, Any, Callable, Optional, Tuple, Iterable


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

DETECTION_LABEL_INDEXES = {
    k.replace("TYPE_", ""):v for k, v in label_pb2.Label.Type.items()
}

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
MAP_POLYLINES_LOCAL_PATH = os.path.join(DATA_PATH, "map_features/polylines")


LOGGER = logging.getLogger(__name__)

os.makedirs(TEMP_DATA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CAMERA_DATA_LOCAL_PATH, exist_ok=True)
os.makedirs(CAMERA_LABELS_LOCAL_PATH, exist_ok=True)
os.makedirs(POINT_CLOUD_LOCAL_PATH, exist_ok=True)
os.makedirs(CAMERA_PROJ_LOCAL_PATH, exist_ok=True)
os.makedirs(LASER_LABELS_LOCAL_PATH, exist_ok=True)
os.makedirs(MAP_POLYLINES_LOCAL_PATH, exist_ok=True)


def download_gcfile_to_temp(gcf_path: str) -> str:
    filename = Path(gcf_path).name
    Gfs.download(gcf_path, TEMP_DATA_PATH)
    temp_path = os.path.join(TEMP_DATA_PATH, filename)
    return temp_path


def transform_map_points(
        polylines: Iterable[map_pb2.MapPoint], 
        transform: np.ndarray,
        xy_min: np.ndarray,
        xy_max: np.ndarray,
    ) -> np.ndarray:

    points = np.asarray([[p.x, p.y, p.z, 1] for p in polylines])
    points = (points @ transform.T)[:, :2]
    mask = (
        (points[:, 0] >= xy_min[0]) & 
        (points[:, 0] <= xy_max[0]) & 
        (points[:, 1] >= xy_min[1]) & 
        (points[:, 1] <= xy_max[1])
    )
    if not np.any(mask):
        return
    points = points[mask]
    return points[:, :2]


def construct_frame_map_elements(
        map_features: dataset_pb2.Frame,
        pose: dataset_pb2.Transform, 
        xy_min: np.ndarray,
        xy_max: np.ndarray,
        eos_token: int,
        pad_token: int, 
    ) -> np.ndarray:
    
    transform = np.linalg.inv(np.reshape(pose.transform, (4, 4)))
    polylines = []
    labels = []

    for feature in map_features:
        if feature.HasField("lane"):
            polyline = transform_map_points(feature.lane.polyline, transform, xy_min, xy_max)
            label = "lane"

        elif feature.HasField("road_line"):
            polyline = transform_map_points(feature.road_line.polyline, transform, xy_min, xy_max)
            label = "road_line"

        elif feature.HasField("road_edge"):
            polyline = transform_map_points(feature.road_edge.polyline, transform, xy_min, xy_max)
            label = "road_edge"

        elif feature.HasField("stop_sign"):
            polyline = transform_map_points([feature.stop_sign.position], transform, xy_min, xy_max)
            label = "stop_sign"

        elif feature.HasField("crosswalk"):
            polyline = transform_map_points(feature.crosswalk.polygon, transform, xy_min, xy_max)
            label = "crosswalk"

        elif feature.HasField("speed_bump"):
            polyline = transform_map_points(feature.speed_bump.polygon, transform, xy_min, xy_max)
            label = "speed_bump"
        
        elif feature.HasField("driveway"):
            polyline = transform_map_points(feature.driveway.polygon, transform, xy_min, xy_max)
            label = "driveway"

        else:
            continue

        if polyline is not None:
            polylines.append(polyline)
            labels.append(MAP_ELEMENT_LABEL_INDEXES[label])
    
    max_vertices = max([p.shape[0] for p in polylines]) + 1
    
    # Let us use eos_token and pad_token to represent end of sequence and padding respectively 
    # respectively for the sake of the VectorMapFormer
    for i in range(0, len(polylines)):
        polyline = polylines[i]
        pads = np.zeros((max_vertices - polyline.shape[0], 2), dtype=polyline.dtype)
        pads.fill(pad_token)
        pads[0, 0] = eos_token
        polylines[i] = np.concatenate([polyline, pads], axis=0)

    polylines = np.stack(polylines, axis=0)
    labels = np.asarray(labels)

    # polyline shape: [label, x_0, y_0, x_1, y_1, x_2, y_2, ..., <eos>, <y_pad>, <x_pad>, <y_pad>]
    polylines = polylines.reshape(polylines.shape[0], -1)
    polylines = np.concatenate([labels[:, None], polylines], axis=1)
    return polylines


def interpolate_points(points: np.ndarray, num_points: int) -> np.ndarray:
    assert points.ndim == 2
    interp_x = np.linspace(0, points.shape[0] - 1, num=num_points)
    orig_x = np.arange(0, points.shape[0])

    interp_points = []
    for i in range(0, points.shape[1]):
        interp_points.append(np.interp(interp_x, orig_x, points[:, i]))
    interp_points = np.stack(interp_points, axis=1)
    return interp_points


def process_perception_frame(
    frame: dataset_pb2.Frame, 
    map_features: Iterable[map_pb2.MapFeature],
    frame_idx: int, 
    sample_name: str, 
    other_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    
    data = {}
    _range_images = {}
    _cam_projections = {}
    _ri_top_pose = None

    camera_views = []
    camera_view_indexes = []
    camera_proj_matrix = {"intrinsic": {}, "extrinsic": {}}
    laser_proj_matrix = {"extrinsic": {}}
    camera_labels = {}
    ego_pose   = np.reshape(frame.pose.transform, (4, 4))
    laser_labels = frame.laser_labels
    timestamp = frame.timestamp_micros

    laser_labels = [{
        "id": label.id,
        "type": label.type,
        "box": {
            "center_x": label.box.center_x, 
            "center_y": label.box.center_y, 
            "center_z": label.box.center_z,
            "length": label.box.length,
            "width": label.box.width,
            "height": label.box.height,
            "heading": label.box.heading
        },
        "metadata": {
            "speed_x": label.metadata.speed_x,
            "speed_y": label.metadata.speed_y,
            "speed_z": label.metadata.speed_z,
            "accel_x": label.metadata.accel_x,
            "accel_y": label.metadata.accel_y,
            "accel_z": label.metadata.accel_z
        },
        "num_top_lidar_points_in_box": label.num_top_lidar_points_in_box,
        "num_lidar_points_in_box": label.num_lidar_points_in_box,
        "most_visible_camera_name": label.most_visible_camera_name,
        "camera_synced_box": "" if not label.most_visible_camera_name else {
            "center_x": label.camera_synced_box.center_x, 
            "center_y": label.camera_synced_box.center_y, 
            "center_z": label.camera_synced_box.center_z,
            "length": label.camera_synced_box.length,
            "width": label.camera_synced_box.width,
            "height": label.camera_synced_box.height,
            "heading": label.camera_synced_box.heading
        },
    } for label in laser_labels]

    for view_idx in range(0, len(frame.images)):
        # images data
        image_data = frame.images[view_idx]
        buffer = np.frombuffer(image_data.image, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (other_kwargs["cam_img_hw"][1], other_kwargs["cam_img_hw"][0]))
        camera_views.append(img)
        camera_view_indexes.append(image_data.name)

        # Laser data
        laser_data = frame.lasers[view_idx]
        if LASER_NAME_INDEX_LABELS[laser_data.name] == "TOP_LIDAR":
            _ri_top_pose = dataset_pb2.MatrixFloat()
            _ri_top_pose.ParseFromString(zlib.decompress(laser_data.ri_return1.range_image_pose_compressed))

        ri1 = zlib.decompress(laser_data.ri_return1.range_image_compressed)
        ri2 = zlib.decompress(laser_data.ri_return2.range_image_compressed)
        cp1 = zlib.decompress(laser_data.ri_return1.camera_projection_compressed)
        cp2 = zlib.decompress(laser_data.ri_return2.camera_projection_compressed)
        
        range_image1 = dataset_pb2.MatrixFloat()
        range_image1.ParseFromString(ri1)
        
        range_image2 = dataset_pb2.MatrixFloat()
        range_image2.ParseFromString(ri2)
        
        cam_projection1 = dataset_pb2.MatrixInt32()
        cam_projection1.ParseFromString(cp1)
        
        cam_projection2 = dataset_pb2.MatrixInt32()
        cam_projection2.ParseFromString(cp2)
        
        _range_images[laser_data.name] = [range_image1, range_image2]
        _cam_projections[laser_data.name] = [cam_projection1, cam_projection2]

        # cam labels for each camera view
        camera_labels[frame.camera_labels[view_idx].name] = [{
            "id": label.id,
            "type": label.type,
            "box": {
                "canter_x": label.box.center_x, 
                "center_y": label.box.center_y, 
                "length": label.box.length, 
                "width": label.box.width
            }
        } for label in frame.camera_labels[view_idx].labels]

        # camera projection matrices
        camera_calibrations = frame.context.camera_calibrations[view_idx]
        cam_intrinsic = np.reshape(camera_calibrations.intrinsic, (3, 3))
        cam_extrinsic = np.reshape(camera_calibrations.extrinsic.transform, (4, 4))
        camera_proj_matrix["intrinsic"][camera_calibrations.name] = cam_intrinsic
        camera_proj_matrix["extrinsic"][camera_calibrations.name] = cam_extrinsic

        # laser projection matrices
        laser_calibrations = frame.context.laser_calibrations[view_idx]
        laser_extrinsic = np.reshape(laser_calibrations.extrinsic.transform, (4, 4))
        laser_proj_matrix["extrinsic"][laser_calibrations.name] = laser_extrinsic

    xy_min = np.asarray([other_kwargs["xy_range"][0][0], other_kwargs["xy_range"][1][0]])
    xy_max = np.asarray([other_kwargs["xy_range"][0][1], other_kwargs["xy_range"][1][1]])

    # camera views
    camera_views = np.stack(camera_views, axis=0)
    camera_views = np.transpose(camera_views, (0, 3, 1, 2))
    camera_view_indexes = np.argsort(camera_view_indexes)
    camera_views = camera_views[camera_view_indexes]
    camera_view_path = os.path.join(CAMERA_DATA_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.npy")
    np.save(camera_view_path, camera_views)

    # map features
    eos_token = other_kwargs["map_elements_eos_token"]
    pad_token = other_kwargs["map_elements_pad_token"]
    other_kwargs = other_kwargs or {}
    map_polylines = construct_frame_map_elements(
        map_features, 
        frame.pose, 
        xy_min=xy_min,
        xy_max=xy_max, 
        eos_token=eos_token,
        pad_token=pad_token,
    )
    map_polylines_path = os.path.join(MAP_POLYLINES_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.npy")
    np.save(map_polylines_path, map_polylines)
        
    # generate LIDAR point cloud data for each LIDAR views
    points, cp_points = convert_range_image_to_point_cloud(
        frame, 
        _range_images, 
        _cam_projections, 
        _ri_top_pose, 
        ri_index=0, 
        keep_polar_features=True
    )

    point_cloud = np.concatenate(points, axis=0)
    camera_proj = np.concatenate(cp_points, axis=0)
    
    points_mask = (point_cloud[:, 3:5] >= xy_min) & (point_cloud[:, 3:5] <= xy_max)
    points_mask = points_mask[:, 0] & points_mask[:, 1]
    
    point_cloud = interpolate_points(point_cloud[points_mask], other_kwargs["num_laser_points"]).astype(np.float32)
    camera_proj = interpolate_points(camera_proj[points_mask], other_kwargs["num_laser_points"]).astype(np.float32)
    
    point_cloud_path = os.path.join(POINT_CLOUD_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.npy")
    np.save(point_cloud_path, point_cloud)

    camera_proj_path = os.path.join(CAMERA_PROJ_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.npy")
    np.save(camera_proj_path, camera_proj)

    camera_labels_path = os.path.join(CAMERA_LABELS_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.pickle")
    laser_labels_path = os.path.join(LASER_LABELS_LOCAL_PATH, f"{sample_name}_frame_{frame_idx}.pickle")
    save_pickle_file(camera_labels, camera_labels_path)
    save_pickle_file(laser_labels, laser_labels_path)

    data[frame_idx] = {
        "camera_view_path": camera_view_path,
        "laser": {
            "point_cloud_path": point_cloud_path,
            "camera_proj_path": camera_proj_path
        },
        "camera_labels_path": camera_labels_path,
        "laser_labels_path": laser_labels_path,
        "ego_pose": ego_pose,
        "camera_proj_matrix": camera_proj_matrix,
        "laser_proj_matrix": laser_proj_matrix,
        "map_elements": {
            "polylines_path": map_polylines_path,
            "eos_token": eos_token,
            "pad_token": pad_token
        },
        "timestamp_seconds": timestamp / 1e6,
    }
    return data


async def offload_to_executor(func: Callable, func_args: Tuple[Any], executor_type: str) -> Any:
    event_loop = asyncio.get_running_loop()
    if executor_type == "process":
        async with ProcessSemaphore:
            return await event_loop.run_in_executor(GlobalProcessPoolExecutor, func, *func_args)
    elif executor_type == "thread":
        async with ThreadSemaphore:
            return await event_loop.run_in_executor(GlobalThreadPoolExecutor, func, *func_args)


async def download_and_process_sample(
        gcf_path: str, 
        dset_category: str,
        other_kwargs: Optional[Dict[str, Any]]=None
    ) -> Tuple[Dict[str, Any], str]:

    data = dict()
    temp_path = await offload_to_executor(
        download_gcfile_to_temp, (gcf_path, ), executor_type="thread"
    )
    dataset = tf.data.TFRecordDataset(temp_path, compression_type="")
    sample_name = Path(temp_path).name
    sample_name = sample_name.replace(".tfrecord", "")
    map_features = None
    tasks = []

    for frame_idx, frame_tensor in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(frame_tensor.numpy()))

        if map_features is None:
            map_features = frame.map_features

        tasks.append(asyncio.create_task(offload_to_executor(
            process_perception_frame, (frame, map_features, frame_idx, sample_name, other_kwargs), executor_type="process"
        )))
        
    frame_dicts = await asyncio.gather(*tasks)
    data[sample_name] = dict()
    for frame_dict in frame_dicts:
        data[sample_name].update(frame_dict)

    await offload_to_executor(delete_path, (temp_path, ), executor_type="thread")
    return data, dset_category


async def run(other_kwargs: Optional[Dict[str, Any]]=None):
    LOGGER.info((
        f"Training size: {len(TrainingFiles)}"
        f"\nTesting size: {len(TestingFiles)}"
        f"\nValidation size: {len(ValidationFiles)}"
    ))
    data = {"data": dict(training=dict(), testing=dict(), validation=dict())}
    tasks = []
    for dset_category, gcf_paths in zip(
        ["training", "testing", "validation"], [TrainingFiles, TestingFiles, ValidationFiles]
    ):
        for gcf_path in gcf_paths: 
            tasks.append(asyncio.create_task(download_and_process_sample(
                gcf_path, dset_category, other_kwargs=other_kwargs
            )))

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)): 
        sample_data, dset_category = await task
        data["data"][dset_category].update(sample_data)

    waymo_metadata = {
        "LABELS": {
            "CAM_NAME_INDEX_LABELS": CAM_NAME_INDEX_LABELS,
            "LASER_NAME_INDEX_LABELS": LASER_NAME_INDEX_LABELS,
            "MAP_ELEMENT_LABEL_INDEXES": MAP_ELEMENT_LABEL_INDEXES,
            "DETECTION_LABEL_INDEXES": DETECTION_LABEL_INDEXES
        },
        "MAP_ELEMENTS":{
            "ROAD_LINE_TYPES": ROAD_LINE_TYPES,
            "ROAD_EDGE_TYPES": ROAD_EDGE_TYPES,
            "LANE_TYPES": LANE_TYPES,
            "CROSSWALK_TYPE": CROSSWALK_TYPE,
            "STOP_SIGN_TYPE": STOP_SIGN_TYPE,
            "DRIVEWAY_TYPE": DRIVEWAY_TYPE,
            "DEFAULT": DEFAULT,
        },
        "PATHS": {
            "CAMERA_DATA_PATH": CAMERA_DATA_LOCAL_PATH,
            "CAMERA_LABELS_PATH": CAMERA_LABELS_LOCAL_PATH,
            "POINT_CLOUD_PATH": POINT_CLOUD_LOCAL_PATH,
            "CAMERA_PROJ_PATH": CAMERA_PROJ_LOCAL_PATH,
            "LASER_LABELS_PATH": LASER_LABELS_LOCAL_PATH,
            "MAP_POLYLINES_PATH": MAP_POLYLINES_LOCAL_PATH
        }
    }

    save_training = asyncio.create_task(offload_to_executor(
        save_pickle_file, data["data"]["training"], os.path.join(DATA_PATH, f"training.pickle"), executor_type="thread"
    ))
    save_testing = asyncio.create_task(offload_to_executor(
        save_pickle_file, data["data"]["testing"], os.path.join(DATA_PATH, f"testing.pickle"), executor_type="thread"
    ))
    save_validation = asyncio.create_task(offload_to_executor(
        save_pickle_file, data["data"]["validation"], os.path.join(DATA_PATH, f"validation.pickle"), executor_type="thread"
    ))
    save_metadata = asyncio.create_task(offload_to_executor(
        save_json_file, waymo_metadata, os.path.join(DATA_PATH, "metadata.json"), executor_type="thread"
    ))
    await asyncio.gather(save_training, save_testing, save_validation, save_metadata)
    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(filename)s: %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description=f"One Time Waymo Dataset Downloader")

    parser.add_argument(
        "--max_thread_workers", type=int, default=20, metavar="", 
        help="Max of workers for thread pool"
    )
    parser.add_argument(
        "--max_proc_workers", type=int, default=20, metavar="", 
        help="Max of workers for process pool"
    )
    parser.add_argument(
        "--thread_concurrency", type=int, default=10, metavar="", 
        help="Number of concurrent thread pool executor calls"
    )
    parser.add_argument(
        "--proc_concurrency", type=int, default=20, metavar="", 
        help="Number of concurrent process pthread pool executor calls"
    )
    parser.add_argument(
        "--cam_img_width", type=int, default=640, metavar="", help="Camera image width"
    )
    parser.add_argument(
        "--cam_img_height", type=int, default=480, metavar="", help="Camera image height"
    )
    parser.add_argument(
        "--num_laser_points", type=int, default=100_000, metavar="", help="Number of points in point cloud"
    )
    parser.add_argument(
        "--map_min_x", type=float, default=-51.2, metavar="", 
        help="minimum value captured by map image along the x-axis"
    )
    parser.add_argument(
        "--map_max_x", type=float, default=51.2, metavar="", 
        help="maximum value captured by map image along the x-axis"
    )
    parser.add_argument(
        "--map_min_y", type=float, default=-51.2, metavar="", 
        help="minimum value captured by map image along the y-axis"
    )
    parser.add_argument(
        "--map_max_y", type=float, default=51.2, metavar="", 
        help="maximum value captured by map image along the y-axis"
    )
    parser.add_argument(
        "--map_elements_eos_token", type=int, default=-998, metavar="", 
        help="eos token for map element polyline vertices"
    )
    parser.add_argument(
        "--map_elements_pad_token", type=int, default=-999, metavar="", 
        help="pad token for map element polyline vertices"
    )

    args = parser.parse_args()

    Gfs = gcsfs.GCSFileSystem(token="google_default")
    TrainingFiles   = Gfs.ls(f"{GCSFS_PERCEPTION_DATA}/training")[1:]
    TestingFiles    = Gfs.ls(f"{GCSFS_PERCEPTION_DATA}/testing")[1:]
    ValidationFiles = Gfs.ls(f"{GCSFS_PERCEPTION_DATA}/validation")[1:]

    mp.set_start_method("spawn", force=True)
    MpContext                 = mp.get_context("spawn")
    ThreadSemaphore           = asyncio.Semaphore(args.thread_concurrency)
    ProcessSemaphore          = asyncio.Semaphore(args.proc_concurrency)
    GlobalThreadPoolExecutor  = ThreadPoolExecutor(args.max_thread_workers)
    GlobalProcessPoolExecutor = ProcessPoolExecutor(args.max_proc_workers, mp_context=MpContext)

    other_kwargs = {
        "cam_img_hw": (args.cam_img_height, args.cam_img_width),
        "xy_range": ((args.map_min_x, args.map_max_x), (args.map_min_y, args.map_max_y)),
        "num_laser_points": args.num_laser_points,
        "map_elements_eos_token": args.map_elements_eos_token,
        "map_elements_pad_token": args.map_elements_pad_token,
    }
    asyncio.run(run(other_kwargs=other_kwargs))
    
    GlobalProcessPoolExecutor.shutdown()
    GlobalThreadPoolExecutor.shutdown()