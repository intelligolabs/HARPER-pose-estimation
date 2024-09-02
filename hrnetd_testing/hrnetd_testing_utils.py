import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple

from lib import dataset
from lib.dataset.harper import HARPER_SKELETON


def compute_depths_and_labels(joint_labels, predicted_pose_2d, depth_frame, fov):
    """
    Compute depths and joint labels based on the predicted 2D pose and depth frame.

    Parameters:
    - joint_labels: Array indicating the visibility of each joint
    - predicted_pose_2d: List of tuples containing the (x, y) coordinates of the predicted 2D pose
    - depth_frame: The depth frame data
    - fov: Field of view

    Returns:
    - depths: List of computed depths for each joint
    - joint_labels: Array indicating the visibility of each joint
    """
    depths = []
    for pidx, (pixel_x, pixel_y) in enumerate(predicted_pose_2d):
        # if the depth frame is empty
        if np.all(depth_frame == 0.0):
            joint_labels[pidx] = 4
            depth = np.median(depths) if depths and np.median(depths) > 0 else 1000
            depths.append(depth)
            continue

        depth = depth_frame[pixel_y, pixel_x]
        stride = 1
        while depth == 0.0:
            joint_labels[pidx] = 1
            fr_y = max(pixel_y - stride, 0)
            to_y = min(pixel_y + stride, depth_frame.shape[0])
            fr_x = max(pixel_x - stride, 0)
            to_x = min(pixel_x + stride, depth_frame.shape[1])
            window = depth_frame[fr_y:to_y, fr_x:to_x]
            depth = (
                np.median(window[window != 0.0]) if not np.all(window == 0.0) else 0.0
            )
            stride += 1
        depths.append(depth)

        if not is_point_in_convex_quadrilateral(pixel_x, pixel_y, fov):
            joint_labels[pidx] = 3

    depths = np.array(depths) / 1000  # convert to meters
    return depths, joint_labels


def adjust_depths(joint_labels, depths, threshold=1.0):
    """
    Adjust depths based on the median of acceptable keypoints.

    Args:
    - joint_labels: Array indicating the visibility of each joint
    - depths: Array of depth values for each joint
    - threshold: Threshold for adjusting depths (default is 1.0)

    Returns:
    - depths: Adjusted depths array
    """
    # Identify acceptable keypoints
    acceptable_keypoints = np.isin(joint_labels, [0, 1])

    if acceptable_keypoints.any():
        # Calculate the median depth of acceptable keypoints
        depth_median = np.median(depths[acceptable_keypoints])

        # Adjust depths that deviate from the median by more than the threshold
        deviations = np.abs(depths - depth_median)
        depths[deviations > threshold] = depth_median

    return depths


def draw_pose(keypoints, img, colors):
    """
    Draw pose keypoints on the image.

    Args:
    - keypoints: List of keypoints to draw
    - img: Image to draw on
    - colors: List of colors for each keypoint

    Returns:
    - None
    """
    for i in range(len(keypoints)):
        x_a, y_a = keypoints[i][0], keypoints[i][1]
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        cv2.circle(img, (int(x_a), int(y_a)), 6, color, -1)


def pixel_to_camera_coordinates(pixel_x, pixel_y, depth, fx, fy, cx, cy):
    """
    Convert pixel coordinates to camera coordinates.

    Args:
    - pixel_x, pixel_y: Pixel coordinates of the point in the 2D frame
    - depth: Depth value at the corresponding pixel location
    - fx, fy: Focal lengths
    - cx, cy: Principal point coordinates

    Returns:
    - X_c, Y_c, Z_c: Camera coordinates of the point
    """
    X_c = (pixel_x - cx) * depth / fx
    Y_c = (pixel_y - cy) * depth / fy
    Z_c = depth
    return X_c, Y_c, Z_c


def camera_to_world_coordinates(X_c, Y_c, Z_c, camera_pose):
    """
    Convert camera coordinates to world coordinates.

    Args:
    - X_c, Y_c, Z_c: Camera coordinates of the point
    - camera_pose: Camera extrinsic parameters (pose) matrix

    Returns:
    - X_w, Y_w, Z_w: World coordinates of the point
    """
    homogeneous_coords = np.vstack([X_c, Y_c, Z_c, np.ones(len(X_c))])
    # from camera to world coordinates
    world_coords = np.dot(np.linalg.inv(camera_pose), homogeneous_coords)
    X_w, Y_w, Z_w = world_coords[0], world_coords[1], world_coords[2]

    return X_w, Y_w, Z_w


def rotate_image(image: np.ndarray, angle: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate an image by a given angle to compensate for the initial camera rotation.

    Args:
    - image: Input image
    - angle: Rotation angle in degrees

    Returns:
    - rotated_image: Rotated image
    - rotation_matrix: Affine transformation matrix
    - top_left: Top-left corner of the rotated image
    """
    center = (image.shape[1] / 2, image.shape[0] / 2)

    # get rotation matrix with translation to include the border
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    # compute new image dimensions
    new_width = int((image.shape[0] * sin_theta) + (image.shape[1] * cos_theta))
    new_height = int((image.shape[0] * cos_theta) + (image.shape[1] * sin_theta))

    # adjust the rotation matrix to include translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # find the first non black pixel to remove the offset
    non_black_pixels = np.where(rotated_image != 0)
    top_left = (np.min(non_black_pixels[1]), np.min(non_black_pixels[0]))
    return rotated_image, rotation_matrix, top_left


def is_point_in_convex_quadrilateral(px, py, vertices):
    """
    Check if a point (px, py) is inside a convex quadrilateral defined by vertices.
    Vertices should be a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
    representing the quadrilateral's vertices in clockwise or counter-clockwise order.

    Args:
    - px, py: Point coordinates
    - vertices: List of vertices of the quadrilateral

    Returns:
    - True if the point is inside the quadrilateral, False otherwise
    """

    def sign(p1, p2, p3):
        return (p3[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p3[1] - p2[1])

    b1 = sign((px, py), vertices[0], vertices[1]) <= 0.0
    b2 = sign((px, py), vertices[1], vertices[2]) <= 0.0
    b3 = sign((px, py), vertices[2], vertices[3]) <= 0.0
    b4 = sign((px, py), vertices[3], vertices[0]) <= 0.0

    return (b1 == b2) and (b2 == b3) and (b3 == b4)


def visualize_3d_pose(axis, pose_3d, color="g", size=5, visibles=None):
    """
    Visualize the 3D pose on a 3D axis.

    Args:
    - axis: 3D axis to plot on
    - pose_3d: 3D pose to visualize
    - color: Color of the pose
    - size: Size of the points
    - visibles: Array indicating the visibility of each joint

    Returns:
    - None
    """
    if visibles is not None:
        pose_3d_vis = pose_3d[visibles]
    else:
        pose_3d_vis = pose_3d

    axis.scatter(
        pose_3d_vis[:, 0],
        pose_3d_vis[:, 1],
        pose_3d_vis[:, 2],
        c=color,
        s=size,
    )
    for i, j in HARPER_SKELETON:
        if visibles is None or (visibles[i] and visibles[j]):
            axis.plot(
                [pose_3d[i, 0], pose_3d[j, 0]],
                [pose_3d[i, 1], pose_3d[j, 1]],
                [pose_3d[i, 2], pose_3d[j, 2]],
                c=color,
            )


def visualize_output(
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
):
    pred_colors = np.array([[0, 0, 255]] * num_joints)
    gt_colors = np.array([[255, 0, 0]] * num_joints)
    max_depth_value = 65535

    # save a copy of the original image
    clean_img = img.copy()

    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324, projection="3d")
    ax4.view_init(90, 0, 90)
    ax4.set_xlim(-1, 2)
    ax4.set_ylim(-0.1, 2)
    ax4.set_zlim(-1, 2)
    ax4.axis("off")
    ax4.grid(False)
    ax5 = fig.add_subplot(325, projection="3d")
    ax5.set_xlim(-1, 2)
    ax5.set_ylim(-0.1, 2)
    ax5.set_zlim(-1, 2)
    ax5.view_init(0, 0, 90)
    ax5.axis("off")
    ax5.grid(False)
    ax6 = fig.add_subplot(326)

    draw_pose(
        np.asarray(gt_human_joints_2d)[visibles_2d],
        img,
        gt_colors[visibles_2d],
    )
    draw_pose(predicted_pose_2d[visibles_2d], img, pred_colors[visibles_2d])

    # draw the predicted pose with joint labels
    label_color_map = {0: [0, 255, 0], 1: [255, 255, 0], 3: [128, 0, 128], 4: [0, 0, 0]}

    label_colors = [label_color_map[l] for l in joint_labels]
    label_colors = np.array(label_colors)

    draw_pose(predicted_pose_2d[visibles_2d], clean_img, label_colors[visibles_2d])

    assert len(visibles_3d) == len(predicted_pose_3d)
    assert len(visibles_3d) == len(joint_labels)

    # draw the depth fov
    for i in range(-1, len(fov) - 1):
        cv2.line(img, tuple(fov[i]), tuple(fov[i + 1]), (255, 0, 0), 2)
        cv2.line(clean_img, tuple(fov[i]), tuple(fov[i + 1]), (255, 0, 0), 2)
        cv2.line(
            depth_frame,
            tuple(fov[i]),
            tuple(fov[i + 1]),
            (max_depth_value, max_depth_value, max_depth_value),
            1,
        )

    ax1.text(0, -5, "red: GT, blue: Predicted", fontsize=10, color="black")
    ax1.imshow(img)
    ax2.text(
        0,
        -5,
        "PREDICTED - Green: ok, Yellow: search again, Purple: out-of-fov, Black: bad depth",
        fontsize=8,
        color="black",
    )
    ax2.imshow(clean_img)
    ax3.imshow(depth_frame * 255 / max_depth_value)

    # visualize the gt 3D pose
    visualize_3d_pose(ax4, gt_human_joints_3d, color="g", size=5)
    visualize_3d_pose(ax5, gt_human_joints_3d, color="g", size=5)

    # visualize the predicted 3D pose
    visualize_3d_pose(ax4, predicted_pose_3d, color="r", size=5, visibles=visibles_3d)
    visualize_3d_pose(ax5, predicted_pose_3d, color="r", size=5, visibles=visibles_3d)

    ax6.imshow(rotated_image)

    # save the plot
    plt.savefig(debug_figure_path)
    plt.close()
