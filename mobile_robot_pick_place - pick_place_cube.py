import pybullet as p
import pybullet_data
import time
import math

# Connect to physics
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load environment
plane = p.loadURDF("plane.urdf")
cube = p.loadURDF("cube_small.urdf", [2.0, 0, 0.05])
# Make the cube red (RGB: 0, 0, 1, Alpha: 1)
p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])

box = p.loadURDF("tray/traybox.urdf", [4.0, 0, 0])

# Load mobile base (Husky)
husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Load Panda arm and attach to Husky
panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.1], useFixedBase=False)
p.createConstraint(
    parentBodyUniqueId=husky,
    parentLinkIndex=-1,
    childBodyUniqueId=panda,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0.2],
    childFramePosition=[0, 0, 0]
)

# Panda indices
end_effector_index = 11
finger_joint_indices = [9, 10]

# Husky wheels
wheel_joints = [2, 3, 4, 5]

def drive_forward(target_x, velocity=5):
    """Drive forward until Husky base reaches target_x coordinate."""
    while True:
        pos, _ = p.getBasePositionAndOrientation(husky)
        if pos[0] >= target_x:
            break
        for i in wheel_joints:
            p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=velocity)
        p.stepSimulation()
        time.sleep(1/240)
    # Stop
    for i in wheel_joints:
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=0)

def drive_backward(distance=0.5, velocity=-3):
    """Drive backward a short distance."""
    pos, orn = p.getBasePositionAndOrientation(husky)
    target_x = pos[0] - distance
    while True:
        pos, _ = p.getBasePositionAndOrientation(husky)
        if pos[0] <= target_x:
            break
        for i in wheel_joints:
            p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=velocity)
        p.stepSimulation()
        time.sleep(1/240)
    # Stop
    for i in wheel_joints:
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=0)

def move_arm(target_pos, target_orn=None):
    """Move Panda arm using IK."""
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([0, -3.14, 0])  # gripper down
    joint_positions = p.calculateInverseKinematics(panda, end_effector_index, target_pos, target_orn)
    for i in range(len(joint_positions)):
        p.setJointMotorControl2(panda, i, p.POSITION_CONTROL, joint_positions[i])
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)

# -------------------------------
# Scripted Sequence
# -------------------------------

# Step 1: Drive near cube (stop ~0.5m before it)
cube_pos, _ = p.getBasePositionAndOrientation(cube)
drive_forward(target_x=cube_pos[0] - 0.7)

# Step 2: Pick cube

move_arm([cube_pos[0], cube_pos[1], 0.2])
move_arm([cube_pos[0], cube_pos[1], 0.02])
cid = p.createConstraint(
    parentBodyUniqueId=panda,
    parentLinkIndex=end_effector_index,
    childBodyUniqueId=cube,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)
move_arm([cube_pos[0], cube_pos[1], 0.3])

# Step 3: Drive near box (stop ~0.7m before it so we don't crash)
box_pos, _ = p.getBasePositionAndOrientation(box)
stop_x = box_pos[0] - 0.7   # adjust until arm can reach safely
drive_forward(target_x=stop_x)


# Step 4: Place cube in box
move_arm([box_pos[0], box_pos[1], 0.3])
move_arm([box_pos[0], box_pos[1], 0.1])
p.removeConstraint(cid)

# Step 5: Retract arm & back away
move_arm([box_pos[0] - 0.2, box_pos[1], 0.4])
drive_backward(distance=0.5)

print("✅ Done! Cube picked, placed in box, and robot moved away.")

# Keep sim running
while True:
    p.stepSimulation()
    time.sleep(1/240)
