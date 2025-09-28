"""
interactive_xarm_final.py

Interactive XArm6 simulation:
- Control each arm joint with safe keys
- Smooth motion while holding keys
- Grasp/release cube
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import math  # At the top, since you're printing Euler angles
import json

# ---------------- CONFIG ----------------
GUI = True
TIME_STEP = 1./240.
CUBE_POS = [0.5, 0.0, 0.05]
CUBE_HALF = 0.03
JOINT_STEP = 0.02  # radians per simulation step
pose_log = {}
pose_count = 0
gripper_state = "open"

# ---------------- INIT ----------------
client = p.connect(p.GUI if GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

# Load XArm6 with gripper
robot = p.loadURDF("xarm/xarm6_with_gripper.urdf", [0, 0, 0], useFixedBase=True)

# Identify revolute joints (arm) and gripper joints
joint_indices = []
gripper_indices = []
print("Joint information:")
for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)
    joint_type = info[2]
    name = info[1].decode('utf-8')
    print(f"  Joint {j}: {name}, type: {joint_type}")
    if joint_type == p.JOINT_REVOLUTE:
        if "finger" in name or "gripper" in name:
            gripper_indices.append(j)
            print(f"    -> Added to gripper joints")
        else:
            joint_indices.append(j)
            print(f"    -> Added to arm joints")

num_joints = len(joint_indices)
print(f"\nFound {num_joints} arm joints and {len(gripper_indices)} gripper joints")
# Alternate colors (red, blue)
colors = [[1, 0, 0, 1], [0, 0, 1, 1]]  # RGBA

for link in range(-1, num_joints):  # -1 is the base link
    color = colors[link % 2]  # alternate red/blue
    p.changeVisualShape(robot, linkIndex=link, rgbaColor=color)
    
if num_joints == 0:
    print("Error: No arm joints found!")
    p.disconnect()
    exit()

ee_link_index = joint_indices[-1] if joint_indices else 0
# ---------------- CREATE TARGET BOX ---------------

# ---------------------------
# Create Hollow Boxes (Containers)
# ---------------------------
def create_hollow_box(pos, size=[0.1, 0.1, 0.05], color=[0.8, 0.8, 0.8, 1]):
    walls = []
    x, y, z = pos
    w, d, h = size
    thickness = 0.01

    floor = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[w, d, thickness]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[w, d, thickness], rgbaColor=color),
        basePosition=[x, y, z - h + thickness]
    )
    walls.append(floor)

    # 4 walls
    for dx, dy in [[w, 0], [-w, 0], [0, d], [0, -d]]:
        wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, d, h] if dx else [w, thickness, h]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, d, h] if dx else [w, thickness, h], rgbaColor=color),
            basePosition=[x + dx, y + dy, z]
        )
        walls.append(wall)

    return walls

container_positions = {
    'cube': [-0.5, -0.3, 0.05]
    #'cylinder': [-0.5, 0.0, 0.05],
    #'sphere': [-0.5, 0.3, 0.05]
}
for name, pos in container_positions.items():
    create_hollow_box(pos)
# ---------------- SAFE KEY MAPPING ----------------
default_keys_inc = ['i','o','p','l','k',';']
default_keys_dec = ['j','u','y','h','n','m']

# Only use as many keys as we have joints
key_increase = [ord(k) for k in default_keys_inc[:num_joints]]
key_decrease = [ord(k) for k in default_keys_dec[:num_joints]]

# Display controls
print("\nControls:")
for idx in range(min(num_joints, len(key_increase), len(key_decrease))):
    print(f"  Joint {idx}: increase={chr(key_increase[idx])} decrease={chr(key_decrease[idx])}")
print("  Gripper: g=close, h=open")
print("  ESC to quit")

# ---------------- CREATE CUBE ----------------
cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3)
cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3, rgbaColor=[1,0.6,0.1,1])
cube_id = p.createMultiBody(0.2, cube_col, cube_vis, CUBE_POS)

grasp_constraint = None
joint_positions = [p.getJointState(robot, j)[0] for j in joint_indices]

print(f"\nStarting simulation with {len(joint_positions)} controllable joints...")

# -------- HELPER FUNCTIONS ----------
def move_to_pose(position, orientation, steps=100):
    for i in range(steps):
        jointPoses = p.calculateInverseKinematics(robot, ee_link_index, position, orientation)
        for j, idx in enumerate(joint_indices):
            p.setJointMotorControl2(robot, idx, p.POSITION_CONTROL, jointPoses[j], force=200)
        p.stepSimulation()
        time.sleep(TIME_STEP)

# ---------------- MAIN LOOP ----------------
while True:
    keys = p.getKeyboardEvents()
    
    # Quit simulation
    if 27 in keys:  # ESC
        break

    # Joint control: smooth motion
    for i in range(len(joint_positions)):  # Use len(joint_positions) instead of num_joints
        if i < len(key_increase) and key_increase[i] in keys:
            joint_positions[i] += JOINT_STEP
        if i < len(key_decrease) and key_decrease[i] in keys:
            joint_positions[i] -= JOINT_STEP

    # Apply joint positions
    for i, j in enumerate(joint_indices):
        if i < len(joint_positions):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joint_positions[i], force=200)

    # Gripper control
    if ord('z') in keys:  # close gripper
        gripper_state = "closed"  # <-- NEW
        for j in gripper_indices:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.0, force=100)
        # grasp cube if EE is close enough
        if joint_indices:  # Make sure we have arm joints
            ee_pos = p.getLinkState(robot, ee_link_index)[0]
            cube_pos = p.getBasePositionAndOrientation(cube_id)[0]
            if np.linalg.norm(np.array(ee_pos)-np.array(cube_pos)) < 0.08 and grasp_constraint is None:
                grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=robot,
                    parentLinkIndex=ee_link_index,
                    childBodyUniqueId=cube_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0,0,0],
                    parentFramePosition=[0,0,0],
                    childFramePosition=[0,0,0]
                )
                print("Cube grasped!")
    
    if ord('x') in keys:  # open gripper
        gripper_state = "open"  # <-- NEW
        for j in gripper_indices:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.04, force=100)
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
            print("Cube released!")

    if ord('1') in keys:
        ee_state = p.getLinkState(robot, ee_link_index)
        pos = ee_state[0]
        orn = ee_state[1]  # quaternion

        pose_name = f"pose_{pose_count}"
        pose_log[pose_name] = {
            "position": [round(x, 4) for x in pos],
            "orientation": [round(x, 4) for x in orn],
            "gripper": gripper_state  # <-- NEW
        }

        pose_count += 1
        print(f"\nSaved {pose_name}: position={pose_log[pose_name]['position']} gripper={gripper_state}")

    if ord('2') in keys:
        break

    if ord('0') in keys:
        try:
            with open("coordinates.json", "r") as f:
                poses = json.load(f)
            print(f"\nLoaded {len(poses)} poses from coordinates.json")

            sorted_keys = sorted(poses.keys(), key=lambda x: int(x.split("_")[1]))

            for name in sorted_keys:
                pos = poses[name]["position"]
                orn = poses[name]["orientation"]
                grip = poses[name].get("gripper", "open")
                print(f"Moving to {name} -> pos: {pos}, gripper: {grip}")
                
                move_to_pose(pos, orn)

                # Set gripper position and apply constraint if needed
                if grip == "closed":
                    gripper_state = "closed"
                    for j in gripper_indices:
                        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.0, force=100)
                    # grasp cube if close enough and not already constrained
                    ee_pos = p.getLinkState(robot, ee_link_index)[0]
                    cube_pos = p.getBasePositionAndOrientation(cube_id)[0]
                    if np.linalg.norm(np.array(ee_pos)-np.array(cube_pos)) < 0.08 and grasp_constraint is None:
                        grasp_constraint = p.createConstraint(
                            parentBodyUniqueId=robot,
                            parentLinkIndex=ee_link_index,
                            childBodyUniqueId=cube_id,
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=[0,0,0],
                            parentFramePosition=[0,0,0],
                            childFramePosition=[0,0,0]
                        )
                        print("Cube grasped during playback!")
                else:
                    gripper_state = "open"
                    for j in gripper_indices:
                        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.04, force=100)
                    if grasp_constraint is not None:
                        p.removeConstraint(grasp_constraint)
                        grasp_constraint = None
                        print("Cube released during playback!")

                time.sleep(0.5)

        except Exception as e:
            print(f"Failed to load or move to poses: {e}")

    # Step simulation
    p.stepSimulation()
    time.sleep(TIME_STEP)

p.disconnect()

if pose_log:
    with open("coordinates.json", "w") as f:
        json.dump(pose_log, f, indent=4)
    print(f"\nSaved {len(pose_log)} poses to coordinates.json")
