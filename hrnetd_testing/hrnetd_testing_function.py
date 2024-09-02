import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple
import pickle
from lib import dataset
from lib.dataset.harper import HARPER_KEYPOINTS, DEPTH_FOVS
from hrnetd_testing_utils import (
    compute_depths_and_labels,
    adjust_depths,
    is_point_in_convex_quadrilateral,
    rotate_image,
    pixel_to_camera_coordinates,
    camera_to_world_coordinates,
    visualize_output,
)


def process_estimated_3d_poses(estimated_3d_poses: List[dict]) -> dict:
    """
    Process the estimated 3D poses to group them by sequence and sort them by frame number.

    Args:
    - estimated_3d_poses: List of estimated 3D poses

    Returns:
    - hrnet_3d_poses: Dictionary containing the estimated 3D poses grouped by sequence
    """
    hrnet_3d_poses = {}

    # group frames by sequence
    frames_by_sequence = defaultdict(list)
    for frame in estimated_3d_poses:
        seq_name = os.path.basename(frame["image_path"])[:-11]
        frames_by_sequence[seq_name].append(frame)

    # process each sequence
    for seq, frames in frames_by_sequence.items():
        hrnet_3d_poses[seq] = [
            {
                "pred_human_joints_2d": frame["pred_human_joints_2d"],
                "pred_human_joints_3d": frame["pred_human_joints_3d"],
                "visibles_2d": frame["visibles_2d"],
                "visibles_3d": frame["visibles_3d"],
                "gt_human_joints_2d": frame["gt_human_joints_2d"],
                "gt_human_joints_3d": frame["gt_human_joints_3d"],
                "depth_fov": frame["depth_fov"],
                "joint_labels": frame["joint_labels"],
                "action": frame["action"],
                "subject": frame["subject"],
                "frame_n": int(frame["frame_n"]),
            }
            for frame in frames
        ]

    # sort the frames by frame number
    for seq in hrnet_3d_poses:
        hrnet_3d_poses[seq] = sorted(hrnet_3d_poses[seq], key=lambda x: x["frame_n"])

    return hrnet_3d_poses


def compute_3d_output(config, output_dir, estimated_3d_poses: List[dict]):
    """
    Compute the 3D output metrics and save the results.

    Args:
    - config: Configuration object
    - output_dir: Output directory
    - estimated_3d_poses: List of estimated 3D poses

    Returns:
    - None
    """
    euc_err = 0
    count_out_of_fov = 0
    count_2d_visibles = 0
    count_3d_visibles = 0
    joints_3d_mpjpe = np.zeros(config.MODEL.NUM_JOINTS)
    joints_3d_counts = np.zeros(config.MODEL.NUM_JOINTS)

    # compute the mpjpe for the visible joints
    for frame in estimated_3d_poses:
        gt_3d_pose = frame["gt_human_joints_3d"]
        pred_3d_pose = frame["pred_human_joints_3d"]
        visibles = frame["visibles_3d"]
        count_out_of_fov += np.sum(frame["visibles_2d"] & (frame["joint_labels"] == 3))
        count_2d_visibles += np.sum(frame["visibles_2d"])
        euc_err += np.sum(
            np.linalg.norm(gt_3d_pose[visibles] - pred_3d_pose[visibles], axis=1)
        )
        count_3d_visibles += np.sum(visibles)
        np.add.at(
            joints_3d_mpjpe,
            visibles,
            np.linalg.norm(gt_3d_pose[visibles] - pred_3d_pose[visibles], axis=1),
        )
        np.add.at(joints_3d_counts, visibles, 1)

    # compute the mpjpe
    mpjpe = euc_err / count_3d_visibles
    print(f"MPJPE: {mpjpe}")
    print(f"3D visibles: {count_3d_visibles} - out of fov: {count_out_of_fov}")
    print(
        f"Percentage of 3D visibles wrt 2D visibles: {round(count_3d_visibles/count_2d_visibles*100, 2)}"
    )

    # mpjpe for each joint
    joints_3d_mpjpe = joints_3d_mpjpe / joints_3d_counts
    print(f"MPJPE for each joint")
    for j, jmpjpe in enumerate(joints_3d_mpjpe):
        print(
            f"Joint {j} ({HARPER_KEYPOINTS[j]}): {round(jmpjpe, 3)} - count: {int(joints_3d_counts[j])}"
        )

    hrnet_3d_poses = process_estimated_3d_poses(estimated_3d_poses)
    # save the 3d poses in pickle format
    with open(
        os.path.join(output_dir, f"hrnetd_poses_harper_{config.DATASET.TEST_SET}.pkl"),
        "wb",
    ) as f:
        pickle.dump(hrnet_3d_poses, f)


def create_3d_projection(config, preds, meta, output_dir, visualize=False):
    """
    Create the 3D projection of the 2D poses.

    Args:
    - config: Configuration object
    - preds: Predicted 2D poses
    - meta: Metadata dictionary
    - output_dir: Output directory
    - visualize: Flag to visualize the 3D projection

    Returns:
    - batch_3d_poses: List of dict with 3D poses
    """
    num_joints = config.MODEL.NUM_JOINTS

    # label the joints
    # 0 - depth value found
    # 1 - depth value not found, increased search window
    # 3 - keypoint out of the depth fov
    # 4 - empty depth frame

    batch_3d_poses = []
    for idx, predicted_pose_2d in enumerate(preds):
        joint_labels = np.zeros(num_joints).astype(int)
        meta_info = {k: v[idx] for k, v in meta.items()}
        img_path = meta_info["image"]
        img = cv2.imread(img_path)
        depth_frame = np.load(meta_info["depth_path"]).astype(np.uint16)

        predicted_pose_2d = predicted_pose_2d.astype(int)
        predicted_pose_3d = []

        visibles_2d = meta_info["gt_human_joints_2d_vis"][:, 0].numpy().astype(bool)
        # compute 3D visibilities
        fov = DEPTH_FOVS[meta_info["rgb_camera_name"]]
        visibles_3d = np.ones(num_joints).astype(bool)
        if (depth_frame == 0.0).all():
            visibles_3d = np.zeros(num_joints).astype(bool)
        else:
            for jdx, (gt_pixel_x, gt_pixel_y) in enumerate(
                meta_info["gt_human_joints_2d"]
            ):
                if not is_point_in_convex_quadrilateral(gt_pixel_x, gt_pixel_y, fov):
                    visibles_3d[jdx] = False

        # camera intrinsics and extrinsics
        intrinsics = np.array(meta_info["intrinsic_rgb"])
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        rvec = np.array(meta_info["extrinsic_rvec"])
        tvec = np.array(meta_info["extrinsic_tvec"])
        rvec = rvec.reshape((3, 3))
        tvec = tvec.reshape((3, 1))
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rvec
        camera_pose[:3, 3] = tvec.flatten()

        depths, joint_labels = compute_depths_and_labels(
            joint_labels, predicted_pose_2d, depth_frame, fov
        )

        depths = adjust_depths(joint_labels, depths)

        rotated_image, rotmat, top_left = rotate_image(
            img, -meta_info["rgb_camera_rotation"].item()
        )

        rotated_image, rotmat, top_left = rotate_image(
            img, -meta_info["rgb_camera_rotation"].item()
        )
        rotated_kpts = cv2.transform(
            predicted_pose_2d.astype(np.float32).reshape(-1, 1, 2), rotmat
        ).reshape(-1, 2)

        # after the rotation some images have a black border, remove it
        if meta_info["rgb_camera_name"] in [
            "frontright_fisheye_image",
            "frontleft_fisheye_image",
        ]:
            rotated_kpts -= top_left

        # convert pixel coordinates to camera and then to world coordinates
        X_c, Y_c, Z_c = pixel_to_camera_coordinates(
            rotated_kpts[:, 0], rotated_kpts[:, 1], depths, fx, fy, cx, cy
        )
        X_w, Y_w, Z_w = camera_to_world_coordinates(X_c, Y_c, Z_c, camera_pose)
        predicted_pose_3d = np.vstack((X_w, Y_w, Z_w)).T.tolist()

        gt_human_joints_2d = np.array(meta_info["gt_human_joints_2d"])
        gt_human_joints_3d = np.array(meta_info["gt_human_joints_3d"])
        predicted_pose_3d = np.array(predicted_pose_3d)
        batch_3d_poses.append(
            {
                "image_path": img_path,
                "pred_human_joints_2d": predicted_pose_2d,
                "pred_human_joints_3d": predicted_pose_3d,
                "gt_human_joints_2d": gt_human_joints_2d,
                "gt_human_joints_3d": gt_human_joints_3d,
                "visibles_2d": visibles_2d,
                "visibles_3d": visibles_3d,
                "depth_fov": fov,
                "joint_labels": joint_labels,
                "action": meta_info["action"],
                "subject": meta_info["subject"],
                "rgb_camera_name": meta_info["rgb_camera_name"],
                "frame_n": meta_info["frame_n"],
            }
        )

        # debug figure
        if visualize and idx % 15 == 0:
            os.makedirs(os.path.join(output_dir, "debug_3D"), exist_ok=True)
            debug_figure_path = os.path.join(
                output_dir, "debug_3D", f"{os.path.basename(img_path)[:-4]}_3D.png"
            )
            visualize_output(
                debug_figure_path,
                img,
                depth_frame,
                predicted_pose_2d,
                predicted_pose_3d,
                visibles_2d,
                visibles_3d,
                gt_human_joints_2d,
                gt_human_joints_3d,
                fov,
                joint_labels,
                rotated_image,
                num_joints,
            )

    return batch_3d_poses
