import pandas as pd
import torch, os
import math

HEIGHT = 1080
WIDTH = 1920
frame_rate = 25
frames = 43

COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face
    (0, 5),  (5, 7), (7, 9), (5, 11),     # left upper body
    (0, 6), (6, 8), (8, 10), (6, 12),     # right upper body
    (11, 13),  (13, 15),                  # Left lowerbody
    (12, 14), (14, 16),                   # Right lowerbody
]

COCO_PARENT = [
    -1,  # 0: Nose (root, no parent)
    0,   # 1: Left Eye ← Nose
    0,   # 2: Right Eye ← Nose
    1,   # 3: Left Ear ← Left Eye
    2,   # 4: Right Ear ← Right Eye
    0,   # 5: Left Shoulder ← Nose (or sometimes Left Ear)
    0,   # 6: Right Shoulder ← Nose (or sometimes Right Ear)
    5,   # 7: Left Elbow ← Left Shoulder
    6,   # 8: Right Elbow ← Right Shoulder
    7,   # 9: Left Wrist ← Left Elbow
    8,   # 10: Right Wrist ← Right Elbow
    5,   # 11: Left Hip ← Left Shoulder (or spine if defined)
    6,   # 12: Right Hip ← Right Shoulder
    11,  # 13: Left Knee ← Left Hip
    12,  # 14: Right Knee ← Right Hip
    13,  # 15: Left Ankle ← Left Knee
    14   # 16: Right Ankle ← Right Knee
]

def angle_difference(a1, a2):
    """Compute wrapped angle difference within [-pi, pi]"""
    delta = a1 - a2
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return delta

def compute_bone_angles(positions, bones):
    """
    Compute 2D joint orientation angles for each bone. (in radian on the image)
    positions: (T, J, 2)
    Returns: (T, num_bones)
    """
    T, J, _ = positions.shape
    angles = []
    for p1_idx, p2_idx in bones:
        vec = positions[:, p2_idx, :] - positions[:, p1_idx, :]  # (T, 2)
        bone_angle = torch.atan2(vec[:, 1], vec[:, 0])     # (T,)
        angles.append(bone_angle)
    return torch.stack(angles, dim=1)  # (T, num_bones)


# Process music beat features into fixed-size vectors, from kimura-san's code
#principle: keep as much resolution as possible, reshape the vector to (L, T//L)
def beat_preprocess(beat_indices, fps, target_fps, start, end):
    # select relevant beat from the precomputed file
    beat_indices = beat_indices[beat_indices >= int(start*fps)]
    beat_indices = beat_indices[beat_indices < int(end*fps)]
    # zero calibrate the indices
    beat_indices = beat_indices - int(start*fps)
    # compute maximum beat dim allowed for current fps
    beat_dim = fps // target_fps # 1 for dance and 4 for music
    if beat_dim == 1:
        beat_dim = 2 #upsample if beat_dim is only 1
    # convert to beat presence vector
    beats_vector = torch.zeros(int(frames*beat_dim))
    # resample the beat_indices (number of new samples / number of og samples)
    scale = frames * beat_dim / ((end-start) * fps)
    beat_indices = torch.round(beat_indices * scale).long()
    beats_vector[beat_indices] = 1 #(along beat_dim, every vector is the same)
    return beats_vector.reshape((frames, int(beat_dim))) #reshape to (L, beat_dim)


def vitpose_preprocess(pose_feature):
    """
    Input: 2d keypoints in COCO format (T, 17, 2)
    Output: [root position, 
            root velocity, 
            joint position (rel to root),
            joint velocity (rel to root),
            joint acceleration (rel to root),
            joint angles (each bone)
            joint angular velocity (each bone)
        ]
    """

    T, J, _ = pose_feature.shape # (T,J,2)
    # using coco format, extract joint position, velocity, acceleration in root space
    root = (pose_feature[:, 11, :] + pose_feature[:, 12, :]) / 2 # (T, 2) midpoint between left and right hip
    root_vel = torch.cat((torch.zeros(1, 2), root[1:] - root[:-1]), 0) #(T, 2)
    joint_pos = pose_feature - root.unsqueeze(1) #(T, J, 2) joint position in root space
    joint_vel = torch.cat((torch.zeros(1, J, 2), joint_pos[1:] - joint_pos[:-1]), 0) #(T, J, 2) joint linear velocity in root space
    joint_acc = torch.cat((torch.zeros(1, J, 2), joint_vel[1:] - joint_vel[:-1]), 0) #(T, J, 2) joint linear acceleration in root space
    # 2d joint orientations, from each parent joint
    # Bone angles on the 2D image
    joint_angles = compute_bone_angles(joint_pos, COCO_BONES)  # (T, num_bones)
    # Bone angular velocity (wrapped between -pi and pi) and pad zeros
    joint_angle_vel = torch.cat((torch.zeros(1, len(COCO_BONES)), angle_difference(joint_angles[1:], joint_angles[:-1])), 0)  # (T, num_bones)
    all_feat = torch.cat((root, # (T, 4+6J+2(J-1)) = (T, 138)
                            root_vel,  
                            joint_pos.flatten(start_dim=1), 
                            joint_vel.flatten(start_dim=1), 
                            joint_acc.flatten(start_dim=1), 
                            joint_angles, 
                            joint_angle_vel), 1)
    
    return all_feat  # shape: (T, 138)


if __name__ == "__main__":
    data_dir = "/data/han_data/dance_data/"
    ann_dir = os.path.join(data_dir, "pdl_chopped_2.csv")
    df = pd.read_csv(ann_dir)
    out_dir = os.path.join(data_dir, "beatdance", "pdl_pose")
    os.makedirs(out_dir, exist_ok=True)
    for i, row in df.iterrows():
        file_id = row["file_id"]
        if not os.path.exists(os.path.join(out_dir, "{}.pt".format(file_id))):
            motion_path = row["motionpath"]
            pose_feature = torch.load(motion_path) # postprocessed motion features (T, J, 2)
            pose_feature = vitpose_preprocess(pose_feature) #extract features (T, 138)
            # preprocess pose 
            torch.save(pose_feature, os.path.join(out_dir, "{}.pt".format(file_id)))

      