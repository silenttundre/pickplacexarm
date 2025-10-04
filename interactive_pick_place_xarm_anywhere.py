# interactive_pick_place_xarm_sorting_safe.py
import pybullet as p
import pybullet_data
import time
import math
import random

# -------------------------------
# Connect & Setup
# -------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("xarm/xarm6_robot.urdf", [0, 0, 0], useFixedBase=True)

# -------------------------------
# Detect joints and EE
# -------------------------------
num_joints = p.getNumJoints(robot_id)
controllable_joints = []
arm_joint_indices = []
gripper_joint_indices = []

GRIPPER_KEYWORDS = ["finger", "gripper", "hand"]

for j in range(num_joints):
    info = p.getJointInfo(robot_id, j)
    jname = info[1].decode("utf-8")
    jtype = info[2]
    if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        controllable_joints.append(j)
        if any(k in jname.lower() for k in GRIPPER_KEYWORDS):
            gripper_joint_indices.append(j)
        else:
            arm_joint_indices.append(j)

ee_link_index = arm_joint_indices[-1] if arm_joint_indices else controllable_joints[-1]

# -------------------------------
# Spawn cubes and trays
# -------------------------------
COLORS = {
    "red": [1, 0, 0, 1],
    "green": [0, 1, 0, 1],
    "blue": [0, 0, 1, 1],
}

# -------------------------------
# Place trays for each color
# -------------------------------
trays = {}
tray_positions = {
    "red":   [0.8, 0.0, 0],   # front
    "green": [-0.6, 0.0, 0],  # back
    "blue":  [0.0, 0.6, 0],   # left side
}

for color, pos in tray_positions.items():
    tray_id = p.loadURDF("tray/traybox.urdf", pos)
    trays[color] = tray_id

# -------------------------------
# Define safe spawn areas (avoid trays)
# -------------------------------
# # Each area: [x_min, x_max, y_min, y_max]
# spawn_areas = [
#     [0.3, 0.6, -0.25, 0.25],   # front-left region
#     [0.3, 0.6, -0.5, -0.3],    # front-right
#     [-0.3, -0.1, -0.25, 0.25], # back-left
#     [-0.3, -0.1, 0.3, 0.5],    # back-right
# ]

# Spawn cubes randomly in safe areas
# -------------------------------
# Spawn cubes safely (avoid trays)
# -------------------------------
cubes = []
num_cubes = 8

# Define the buffer around trays (x,y radius)
tray_buffer = 0.15

while len(cubes) < num_cubes:
    color = random.choice(list(COLORS.keys()))
    # Random position in general spawn area
    cube_x = random.uniform(-0.6, 0.6)
    cube_y = random.uniform(-0.6, 0.6)

    # Check if inside any tray + buffer
    safe_to_spawn = True
    for t_color, t_id in trays.items():
        tray_pos, _ = p.getBasePositionAndOrientation(t_id)
        if (abs(cube_x - tray_pos[0]) < tray_buffer and
            abs(cube_y - tray_pos[1]) < tray_buffer):
            safe_to_spawn = False
            break

    if safe_to_spawn:
        cube_id = p.loadURDF("cube_small.urdf", [cube_x, cube_y, 0.05])
        p.changeVisualShape(cube_id, -1, rgbaColor=COLORS[color])
        cubes.append((cube_id, color))

# -------------------------------
# Motion helpers
# -------------------------------
def move_to_pose(target_pos, target_orn, steps=200):
    ik_solution = p.calculateInverseKinematics(robot_id, ee_link_index, target_pos, target_orn)
    for i, joint_idx in enumerate(controllable_joints):
        if i < len(ik_solution):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                    targetPosition=ik_solution[i], force=200)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)

def open_gripper():
    for j in gripper_joint_indices:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=100)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

def close_gripper():
    for j in gripper_joint_indices:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=200)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

# -------------------------------
# Pick & place routines
# -------------------------------
def pick_up_cube(cube_id):
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    target_orn = p.getQuaternionFromEuler([0, math.pi, 0])

    # Hover above cube
    move_to_pose([cube_pos[0], cube_pos[1], cube_pos[2] + 0.2], target_orn, 250)
    open_gripper()

    # Lower to cube
    move_to_pose([cube_pos[0], cube_pos[1], cube_pos[2] + 0.02], target_orn, 220)
    close_gripper()

    # Attach constraint
    cid = p.createConstraint(robot_id, ee_link_index, cube_id, -1,
                             p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

    # Lift up
    move_to_pose([cube_pos[0], cube_pos[1], cube_pos[2] + 0.3], target_orn, 250)
    return cid

def place_cube(constraint_id, cube_id, tray_id):
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
    target_orn = p.getQuaternionFromEuler([0, math.pi, 0])

    # Safe hover above tray
    safe_hover = [tray_pos[0] - 0.1, tray_pos[1], tray_pos[2] + 0.35]
    move_to_pose(safe_hover, target_orn, 300)

    # Slowly lower with a small random offset inside tray
    offset_x = random.uniform(-0.03, 0.03)
    offset_y = random.uniform(-0.03, 0.03)
    target_hover = [tray_pos[0] + offset_x, tray_pos[1] + offset_y, tray_pos[2] + 0.12]
    move_to_pose(target_hover, target_orn, 250)

    # Release cube
    p.removeConstraint(constraint_id)
    open_gripper()

    # Retract safely
    move_to_pose([tray_pos[0], tray_pos[1], tray_pos[2] + 0.4], target_orn, 250)

# -------------------------------
# Main Loop: Sort cubes
# -------------------------------
for cube_id, color in cubes:
    print(f"Sorting cube {cube_id} ({color})...")
    cid = pick_up_cube(cube_id)
    place_cube(cid, cube_id, trays[color])

print("✅ All cubes sorted safely!")

# Keep simulation running
while True:
    p.stepSimulation()
    time.sleep(1./240.)
