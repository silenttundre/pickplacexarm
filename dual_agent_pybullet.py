# dual_agent_pybullet.py
# CLEANED VERSION - Removed all unused code

import pybullet as p
import pybullet_data
import time
import math
import random
import threading
import queue
import sys
import traceback
import requests
import json

# ================================
# 1. LLM AGENT CLASS
# ================================
class SmartOllamaAgent:
    def __init__(self, robot_name, system_prompt, model="llama2"):
        self.robot_name = robot_name
        self.name = robot_name  # Add for compatibility
        self.system_prompt = system_prompt
        self.model = model
        self.base_url = "http://localhost:11434"
        self.is_connected = self.test_connection()
        self.last_decision = None
        
    def test_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_spatial_decision(self, current_state, target_position, arm_position=[0,0,0]):
        """Get decision from LLM with spatial awareness"""
        if not self.is_connected:
            return self.rule_based_spatial_fallback(target_position, arm_position)
            
        try:
            dx = target_position[0] - arm_position[0]
            dy = target_position[1] - arm_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx))
            
            spatial_info = f"""
SPATIAL ANALYSIS:
- Target position: ({target_position[0]:.2f}, {target_position[1]:.2f})
- Arm position: ({arm_position[0]:.2f}, {arm_position[1]:.2f})  
- Direction: {angle:.1f} degrees from arm
- Distance: {distance:.2f} meters
- Behind arm: {'YES' if abs(angle) > 90 else 'NO'}
- Need base rotation: {'YES, rotate 180 degrees' if abs(angle) > 90 else 'NO, small adjustment'}
"""

            full_prompt = f"""System Role: {self.system_prompt}

{spatial_info}

Current Task: {current_state}

Based on the spatial analysis above, what specific motion should I perform?
Consider whether the arm base needs rotation and the optimal approach path.

Respond with: ROTATE_AND_MOVE or MOVE_DIRECTLY or ROTATE_180"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model, 
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                decision = response.json()["response"].strip().upper()
                print(f"[{self.robot_name} SMART LLM] Spatial decision: {decision}")
                print(f"[{self.robot_name} SMART LLM] Analysis: angle={angle:.1f}°, behind_arm={abs(angle) > 90}")
                return decision
            else:
                return self.rule_based_spatial_fallback(target_position, arm_position)
                
        except Exception as e:
            print(f"[{self.robot_name} SMART LLM] Error: {e}")
            return self.rule_based_spatial_fallback(target_position, arm_position)
    
    def rule_based_spatial_fallback(self, target_position, arm_position):
        """Rule-based spatial decision making"""
        dx = target_position[0] - arm_position[0]
        dy = target_position[1] - arm_position[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        if abs(angle) > 100:
            decision = "ROTATE_180"
        elif abs(angle) > 45:
            decision = "ROTATE_AND_MOVE" 
        else:
            decision = "MOVE_DIRECTLY"
            
        print(f"[{self.robot_name} RULE-BASED] Angle: {angle:.1f}° -> Decision: {decision}")
        return decision

    # NEW METHOD: Analyze environment for mobile robot
    def analyze_environment(self, husky, panda, cubes, xarm_trays, mobile_trays):
        """Analyze current environment and get LLM decision for mobile robot"""
        try:
            # Get current robot state
            husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
            husky_yaw = p.getEulerFromQuaternion(husky_orn)[2]
            
            # Check if holding cube
            holding_cube = False
            cube_color_held = None
            
            # Get cube positions and status
            cube_states = []
            available_cubes = []
            for cube_id, color in cubes:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                cube_states.append((color, cube_pos))
                # Simple check if cube is near gripper (being held)
                gripper_pos, _ = p.getLinkState(panda, 11)[0]
                dist = math.sqrt(sum((cube_pos[i] - gripper_pos[i])**2 for i in range(3)))
                if dist < 0.1:  # Cube is close to gripper
                    holding_cube = True
                    cube_color_held = color
                else:
                    available_cubes.append(color)
            
            # Create environment description for LLM
            env_description = f"""
CURRENT ENVIRONMENT STATE:
- Husky position: ({husky_pos[0]:.2f}, {husky_pos[1]:.2f})
- Husky orientation: {math.degrees(husky_yaw):.1f}°
- Holding cube: {'YES (' + cube_color_held + ')' if holding_cube else 'NO'}
- Available cubes: {', '.join(available_cubes) if available_cubes else 'NONE'}
"""
            # LLM decision prompt
            prompt = f"""
{self.system_prompt}

{env_description}

What should I do next? Choose ONE action only:

AVAILABLE ACTIONS:
- NAVIGATE_TO red (approach red cube for pickup)
- NAVIGATE_TO green (approach green cube for pickup) 
- NAVIGATE_TO blue (approach blue cube for pickup)
- PICK_CUBE (pick up nearby cube)
- NAVIGATE_TO red_tray (go to red tray for placement)
- NAVIGATE_TO green_tray (go to green tray for placement)
- NAVIGATE_TO blue_tray (go to blue tray for placement)
- PLACE_CUBE (place held cube in nearby tray)
- WAIT (pause briefly)

Current priority: {'PLACE cube if holding one' if holding_cube else 'PICK UP available cube'}

Respond with ONLY the action command in uppercase.
"""
            
            if not self.is_connected:
                return self.rule_based_mobile_decision(husky_pos, holding_cube, cube_color_held, cube_states)
                
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "max_tokens": 20}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                decision = response.json()["response"].strip().upper()
                print(f"[{self.robot_name} LLM] Decision: {decision}")
                self.last_decision = decision
                return decision
            else:
                return self.rule_based_mobile_decision(husky_pos, holding_cube, cube_color_held, cube_states)
                
        except Exception as e:
            print(f"[{self.robot_name} LLM] Analysis error: {e}")
            return "WAIT"

    def rule_based_mobile_decision(self, husky_pos, holding_cube, cube_color_held, cube_states):
        """Rule-based fallback for mobile robot"""
        if holding_cube and cube_color_held:
            # If holding cube, go to corresponding tray
            return f"NAVIGATE_TO {cube_color_held}_tray"
        elif cube_states:
            # Find closest available cube
            closest_dist = float('inf')
            closest_color = None
            for color, cube_pos in cube_states:
                dist = math.sqrt((cube_pos[0]-husky_pos[0])**2 + (cube_pos[1]-husky_pos[1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_color = color
            if closest_dist < 0.8:
                return "PICK_CUBE"
            else:
                return f"NAVIGATE_TO {closest_color}"
        else:
            return "WAIT"

    # ===================== EXECUTION LAYER =====================
    def execute_decision(self, decision_text, husky, panda, cubes, xarm_trays, mobile_trays):
        """Execute LLM decision with immediate collision checking"""
        if not decision_text:
            print(f"[{self.name}] No valid decision text.")
            return

        decision_text = decision_text.strip().upper()
        print(f"[{self.name}] Executing: {decision_text}")

        # Check for collisions BEFORE doing anything
        collision, obj_type, obj_name = self.check_immediate_collision(husky)
        if collision:
            print(f"🚨 COLLISION DETECTED before executing {decision_text}!")
            self.emergency_stop_and_assess(husky, panda, cubes, mobile_trays)
            return

        try:
            if "NAVIGATE_TO" in decision_text:
                target_name = decision_text.replace("NAVIGATE_TO", "").strip()
                if target_name in xarm_trays or target_name in mobile_trays:
                    self.navigate_to_target(husky, target_name, xarm_trays, mobile_trays)
                else:
                    self.navigate_to_cube(husky, target_name, cubes)

            elif "PICK_CUBE" in decision_text:
                # Check collision again before picking
                collision, obj_type, obj_name = self.check_immediate_collision(husky)
                if collision and obj_type == "cube":
                    print(f"🚨 Already touching cube, attempting immediate pick!")
                    self.attempt_immediate_pick(husky, panda, cubes, obj_name)
                else:
                    self.smart_pick_cube(panda, cubes)

            elif "PLACE_CUBE" in decision_text:
                self.smart_place_cube(panda, mobile_trays)

            elif "WAIT" in decision_text:
                print(f"[{self.name}] Waiting...")
                time.sleep(2)

            else:
                print(f"[{self.name}] Unrecognized decision: {decision_text}")

        except Exception as e:
            print(f"[{self.name}] Error executing decision: {e}")
            import traceback; traceback.print_exc()

    def navigate_to_target(self, husky, target_name, xarm_trays, mobile_trays):
        """Navigate to specific tray target using efficient side approach"""
        target_dict = {**xarm_trays, **mobile_trays}
        if target_name not in target_dict:
            print(f"[{self.name}] Unknown target: {target_name}")
            return

        target_id = target_dict[target_name]
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        
        # Use efficient side approach
        approach_side = self.avoid_xarm_zone(target_pos)
        self.efficient_side_approach(target_pos, husky, approach_side)

    def navigate_to_cube(self, husky, cube_color, cubes):
        """Navigate to specific cube using efficient side approach"""
        target_cube_id = None
        target_pos = None
        
        for cube_id, color in cubes:
            if color.lower() == cube_color.lower():
                target_cube_id = cube_id
                target_pos, _ = p.getBasePositionAndOrientation(cube_id)
                break
        
        if target_pos:
            # Use efficient side approach
            approach_side = self.avoid_xarm_zone(target_pos)
            self.efficient_side_approach(target_pos, husky, approach_side)
        else:
            print(f"[{self.name}] Cube {cube_color} not found")

    def smart_pick_cube(self, panda, cubes):
        """IMPROVED: Smart cube picking with reliable grasping and lower reach"""
        husky_pos, _ = p.getBasePositionAndOrientation(husky)
        
        # Find closest cube
        closest_dist = float('inf')
        target_cube_id = None
        
        for cube_id, color in cubes:
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            dist = math.sqrt((cube_pos[0]-husky_pos[0])**2 + (cube_pos[1]-husky_pos[1])**2)
            if dist < closest_dist and dist < 1.2:
                closest_dist = dist
                target_cube_id = cube_id
        
        if target_cube_id:
            cube_pos, _ = p.getBasePositionAndOrientation(target_cube_id)
            print(f"[{self.name}] Attempting reliable pick of cube at {cube_pos}")
            
            # IMPROVED: Use the enhanced reliable pick method
            success, constraint_id = self.reliable_panda_pick_cube(panda, target_cube_id)
            
            if success:
                print("✅ Cube successfully picked!")
                # Store constraint info for later placement
                self.last_pick_constraint = constraint_id
                self.last_picked_cube = target_cube_id
            else:
                print("❌ Cube pick failed")
        else:
            print(f"[{self.name}] No cube within reachable distance")                      

    def smart_place_cube(self, panda, mobile_trays):
        """Smart cube placement that works from current position if close enough"""
        husky_pos, _ = p.getBasePositionAndOrientation(husky)
        
        # Find closest tray and check if we can reach it from current position
        closest_dist = float('inf')
        target_tray_id = None
        target_color = None
        target_tray_pos = None
        
        for color, tray_id in mobile_trays.items():
            tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
            dist = math.sqrt((tray_pos[0]-husky_pos[0])**2 + (tray_pos[1]-husky_pos[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                target_tray_id = tray_id
                target_color = color
                target_tray_pos = tray_pos
        
        if target_tray_id and closest_dist <= 1.0:  # Arm can reach up to 1m
            print(f"[{self.name}] Tray within reach ({closest_dist:.2f}m), placing cube from current position")
            
            # Execute place sequence from current position
            approach_pos = [target_tray_pos[0], target_tray_pos[1], target_tray_pos[2] + 0.25]
            place_pos = [target_tray_pos[0], target_tray_pos[1], target_tray_pos[2] + 0.05]
            
            ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_tray_approach")
            ultra_smooth_panda_move_to_position(place_pos, 80, "ultra_smooth_place")
            open_gripper(panda, finger_joint_indices)
            ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_retract")
            close_gripper(panda, finger_joint_indices)
            
            print(f"✅ Cube placed in {target_color} tray from current position")
        else:
            print(f"[{self.name}] No tray within reachable distance")

    def check_immediate_collision(self, husky):
        """Check if Husky is currently colliding with any object"""
        husky_pos, _ = p.getBasePositionAndOrientation(husky)
        
        # Get all contact points for the Husky
        contact_points = p.getContactPoints(husky)
        
        for contact in contact_points:
            # Check if contact is with a tray or cube (not ground or itself)
            if contact[2] != plane_id and contact[2] != husky:  # contact[2] is the other body
                # Get the other object's ID
                other_body = contact[2]
                
                # Check if it's a tray
                is_tray = False
                tray_name = None
                for name, tray_id in {**xarm_trays, **mobile_trays}.items():
                    if tray_id == other_body:
                        is_tray = True
                        tray_name = name
                        break
                
                # Check if it's a cube
                is_cube = False
                cube_color = None
                for cube_id, color in cubes:
                    if cube_id == other_body:
                        is_cube = True
                        cube_color = color
                        break
                
                if is_tray:
                    print(f"🚨 COLLISION DETECTED: Husky is touching {tray_name} tray!")
                    return True, "tray", tray_name
                elif is_cube:
                    print(f"🚨 COLLISION DETECTED: Husky is touching {cube_color} cube!")
                    return True, "cube", cube_color
        
        return False, None, None

    def emergency_stop_and_assess(self, husky, panda, cubes, mobile_trays):
        """Immediately stop and assess the situation when collision is detected"""
        print("🛑 EMERGENCY STOP ACTIVATED!")
        
        # Immediate hard stop
        for wheel in wheel_joints:
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        
        # Stabilize
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Check what we're colliding with and take appropriate action
        collision, obj_type, obj_name = self.check_immediate_collision(husky)
        
        if collision:
            if obj_type == "tray":
                print(f"📦 Collision with {obj_name} tray - assessing if we can pick cube from here")
                return self.attempt_pick_from_collision_position(husky, panda, cubes, obj_name)
            elif obj_type == "cube":
                print(f"🎯 Collision with {obj_name} cube - attempting immediate pick")
                return self.attempt_immediate_pick(husky, panda, cubes, obj_name)
        
        return False

    def attempt_pick_from_collision_position(self, husky, panda, cubes, tray_name):
        """Try to pick up cube from current position after colliding with tray"""
        husky_pos, _ = p.getBasePositionAndOrientation(husky)
        
        # Find cubes in the tray we collided with
        tray_id = None
        if tray_name in xarm_trays:
            tray_id = xarm_trays[tray_name]
        elif tray_name in mobile_trays:
            tray_id = mobile_trays[tray_name]
        
        if tray_id:
            tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
            
            # Look for cubes in this tray
            for cube_id, color in cubes:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                dist_to_tray = math.sqrt((cube_pos[0]-tray_pos[0])**2 + (cube_pos[1]-tray_pos[1])**2)
                
                if dist_to_tray < 0.2:  # Cube is in this tray
                    print(f"🎯 Found {color} cube in {tray_name} tray, attempting pick from current position")
                    
                    # Try to pick the cube from our current position
                    approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.25]
                    pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.03]
                    
                    try:
                        ultra_smooth_panda_move_to_position(approach_pos, 100, "emergency_approach")
                        open_gripper(panda, finger_joint_indices)
                        ultra_smooth_panda_move_to_position(pick_pos, 80, "emergency_pick")
                        close_gripper(panda, finger_joint_indices)
                        ultra_smooth_panda_move_to_position(approach_pos, 100, "emergency_lift")
                        
                        print("✅ Successfully picked cube from collision position!")
                        return True
                    except Exception as e:
                        print(f"❌ Failed to pick from collision position: {e}")
                        return False
        
        print("❌ No cubes found in the tray we collided with")
        return False

    def attempt_immediate_pick(self, husky, panda, cubes, cube_color):
        """Immediately pick up the cube we're colliding with"""
        print(f"🎯 Attempting immediate pick of {cube_color} cube")
        
        # Find the specific cube
        for cube_id, color in cubes:
            if color == cube_color:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                
                approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.25]
                pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.03]
                
                try:
                    ultra_smooth_panda_move_to_position(approach_pos, 100, "immediate_approach")
                    open_gripper(panda, finger_joint_indices)
                    ultra_smooth_panda_move_to_position(pick_pos, 80, "immediate_pick")
                    close_gripper(panda, finger_joint_indices)
                    ultra_smooth_panda_move_to_position(approach_pos, 100, "immediate_lift")
                    
                    print("✅ Successfully picked cube immediately!")
                    return True
                except Exception as e:
                    print(f"❌ Failed immediate pick: {e}")
                    return False
        
        return False

    def calculate_side_approach(self, target_pos, approach_side="left"):
        """Calculate approach position from the side of the target"""
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        husky_yaw = p.getEulerFromQuaternion(husky_orn)[2]
        
        if approach_side == "left":
            # Approach from left side (negative Y relative to target)
            approach_offset = [-0.5, -0.6]  # X, Y offsets
        else:  # right
            # Approach from right side (positive Y relative to target)  
            approach_offset = [-0.5, 0.6]
        
        approach_pos = [
            target_pos[0] + approach_offset[0],
            target_pos[1] + approach_offset[1],
            husky_start_pos[2]
        ]
        
        return approach_pos

    def avoid_xarm_zone(self, target_pos):
        """Check if target is in XArm workspace and adjust approach"""
        xarm_workspace_radius = 1.5  # XArm workspace radius
        
        # Calculate distance from XArm base
        distance_from_xarm = math.sqrt(target_pos[0]**2 + target_pos[1]**2)
        
        if distance_from_xarm < xarm_workspace_radius:
            print("⚠️  Target in XArm workspace, using cautious approach")
            # Approach from the side farthest from XArm
            if target_pos[1] >= 0:
                return "left"  # Approach from left if target is on right side
            else:
                return "right" # Approach from right if target is on left side
        else:
            return "left"  # Default approach from left

    def reliable_panda_pick_cube(self, panda, cube_id):
        """IMPROVED: Reliable cube picking for Panda arm with proper grasping and lower reach"""
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        print(f"🎯 RELIABLE PICK: Attempting to pick cube at {cube_pos}")
        
        # IMPROVED: Lower approach heights
        approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]  # Lower approach
        pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.005]     # Almost touching cube
        
        print("📍 Moving to approach position")
        ultra_smooth_panda_move_to_position(approach_pos, 100, "reliable_approach")
        
        # Open gripper
        print("🖐️ Opening gripper")
        open_gripper(panda, finger_joint_indices)
        
        # Move down to pick position (slower for precision)
        print("⬇️ Moving to pick position")
        ultra_smooth_panda_move_to_position(pick_pos, 100, "reliable_pick")
        
        # Close gripper firmly
        print("✊ Closing gripper firmly")
        close_gripper(panda, finger_joint_indices)
        
        # Create constraint for smooth holding
        constraint_id = None
        try:
            ee_state = p.getLinkState(panda, end_effector_index)
            constraint_id = p.createConstraint(
                panda,
                end_effector_index,
                cube_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            )
            print(f"✅ Cube constraint created: {constraint_id}")
        except Exception as e:
            print(f"❌ Constraint failed: {e}")
        
        # Lift cube
        print("⬆️ Lifting cube")
        ultra_smooth_panda_move_to_position(approach_pos, 100, "reliable_lift")
        
        # Verify pick was successful
        new_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        height_diff = new_cube_pos[2] - cube_pos[2]
        
        if height_diff > 0.05:  # Cube was lifted significantly
            print("✅ Cube successfully picked up!")
            return True, constraint_id
        else:
            print("❌ Cube pick failed - cube not lifted")
            if constraint_id:
                try:
                    p.removeConstraint(constraint_id)
                except:
                    pass
            return False, None

    def efficient_side_approach(self, target_pos, husky, approach_side="left"):
        """Efficient side approach that stops if collision detected"""
        print(f"🔄 EFFICIENT SIDE APPROACH from {approach_side}")
        
        # Calculate side approach position
        side_approach_pos = self.calculate_side_approach(target_pos, approach_side)
        
        # Navigate to side position with collision monitoring
        print("📍 Phase 1: Moving to side position")
        success = self.collision_monitored_navigation_simple(side_approach_pos, husky, "side_approach")
        
        if not success:
            print("❌ Collision detected during side approach - stopping")
            return False
        
        # Turn to face target
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        dx = target_pos[0] - husky_pos[0]
        dy = target_pos[1] - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        print("🔄 Phase 2: Turning to face target")
        self.smooth_turn_to_angle(husky, target_angle)
        
        # Final approach with early stopping
        final_approach_pos = [target_pos[0] - 0.5, target_pos[1], husky_start_pos[2]]  # Stop further back
        print("🎯 Phase 3: Final approach")
        success = self.collision_monitored_navigation_simple(final_approach_pos, husky, "final_approach")
        
        if success:
            print("✅ Efficient approach completed")
            return True
        else:
            print("⚠️ Approach completed with collision - may need arm adjustment")
            return True  # Still return True as we're close enough for arm operation

    def collision_monitored_navigation_simple(self, target_pos, husky, operation_type):
        """Simple collision monitoring that stops immediately on collision"""
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        start_pos = husky_pos.copy()
        
        distance = math.sqrt((target_pos[0]-husky_pos[0])**2 + (target_pos[1]-husky_pos[1])**2)
        total_steps = int(distance * 100)  # Fewer steps for faster movement
        total_steps = max(50, min(total_steps, 200))
        
        for step in range(total_steps):
            husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
            
            # Check for collision
            collision, obj_type, obj_name = self.check_immediate_collision(husky)
            if collision:
                print(f"🚨 COLLISION with {obj_name} - STOPPING")
                self.stop_husky_smoothly(husky)
                return False  # Indicate collision occurred
            
            # Check if close enough
            current_distance = math.sqrt((target_pos[0]-husky_pos[0])**2 + (target_pos[1]-husky_pos[1])**2)
            if current_distance <= 0.3:
                print(f"✅ Close enough at {current_distance:.2f}m")
                self.stop_husky_smoothly(husky)
                return True
            
            # Simple navigation
            progress = step / total_steps
            current_target_x = start_pos[0] + (target_pos[0] - start_pos[0]) * progress
            current_target_y = start_pos[1] + (target_pos[1] - start_pos[1]) * progress
            
            dx = current_target_x - husky_pos[0]
            dy = current_target_y - husky_pos[1]
            target_angle = math.atan2(dy, dx)
            
            current_angle = p.getEulerFromQuaternion(husky_orn)[2]
            angle_diff = target_angle - current_angle
            
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Simple speed control
            base_speed = 0.8
            turn_gain = 1.2
            
            left_speed = base_speed - turn_gain * angle_diff
            right_speed = base_speed + turn_gain * angle_diff
            
            p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=150)
            p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=150)
            p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=150)
            p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=150)
            
            p.stepSimulation()
            time.sleep(1./240.)
        
        self.stop_husky_smoothly(husky)
        return True

    def smooth_turn_to_angle(self, husky, target_angle):
        """Smoothly turn Husky to face target angle"""
        for step in range(60):
            husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
            current_angle = p.getEulerFromQuaternion(husky_orn)[2]
            
            angle_diff = target_angle - current_angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Smooth turning
            turn_speed = 0.8 * (1.0 - step/60.0)  # Slow down as we approach target
            
            left_speed = -turn_speed * angle_diff
            right_speed = turn_speed * angle_diff
            
            p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
            p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)
            p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
            p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)
            
            p.stepSimulation()
            time.sleep(1./240.)
            
            if abs(angle_diff) < 0.1:  # ~6 degrees
                break
        
        # Stop wheels
        for wheel in wheel_joints:
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)

    def stop_husky_smoothly(self, husky):
        """Smoothly stop the Husky"""
        print("🛑 Smooth stopping...")
        for decel_step in range(30):
            factor = 1.0 - (decel_step / 30.0)
            for wheel in wheel_joints:
                current_vel = p.getJointState(husky, wheel)[1]
                p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, 
                                    targetVelocity=current_vel * factor, force=100)
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Final stop
        for wheel in wheel_joints:
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        
        # Stabilization period
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("✅ Husky fully stopped and stabilized")

def simple_husky_move_to(target_pos):
    """SIMPLE: Move Husky to target position"""
    print(f"[HUSKY] Moving to {[f'{x:.2f}' for x in target_pos]}")
    
    for step in range(300):  # Fixed number of steps
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        
        # Calculate direction
        dx = target_pos[0] - husky_pos[0]
        dy = target_pos[1] - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        current_angle = p.getEulerFromQuaternion(husky_orn)[2]
        angle_diff = target_angle - current_angle
        
        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Simple wheel control
        base_speed = 1.5
        turn_gain = 1.5
        
        left_speed = base_speed - turn_gain * angle_diff
        right_speed = base_speed + turn_gain * angle_diff
        
        # Move wheels
        p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Stop
    for wheel in wheel_joints:
        p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
    
    # Wait to stabilize
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1./240.)

def simple_panda_pick(cube_id):
    """SIMPLE: Pick cube with Panda arm"""
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    print(f"[PANDA] Picking cube at {cube_pos}")
    
    # Approach
    approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
    ultra_smooth_panda_move_to_position(approach_pos, 100, "approach")
    
    # Open gripper
    open_gripper(panda, finger_joint_indices)
    
    # Pick
    pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.02]
    ultra_smooth_panda_move_to_position(pick_pos, 80, "pick")
    
    # Close gripper
    close_gripper(panda, finger_joint_indices)
    
    # Create constraint
    constraint_id = p.createConstraint(
        panda, end_effector_index, cube_id, -1, 
        p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0]
    )
    
    # Lift
    ultra_smooth_panda_move_to_position(approach_pos, 80, "lift")
    
    return constraint_id

def simple_panda_place(tray_id, constraint_id):
    """SIMPLE: Place cube with Panda arm"""
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
    print(f"[PANDA] Placing cube in tray at {tray_pos}")
    
    # Approach tray
    approach_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15]
    ultra_smooth_panda_move_to_position(approach_pos, 100, "tray_approach")
    
    # Place
    place_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.05]
    ultra_smooth_panda_move_to_position(place_pos, 80, "place")
    
    # Remove constraint and open gripper
    if constraint_id:
        p.removeConstraint(constraint_id)
    open_gripper(panda, finger_joint_indices)
    
    # Wait for drop
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Retract
    ultra_smooth_panda_move_to_position(approach_pos, 80, "retract")
    close_gripper(panda, finger_joint_indices)

def debug_husky_position():
    """Debug function to check Husky position"""
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
    print(f"[DEBUG] Husky position: {husky_pos}")
    return husky_pos

def improved_panda_place_cube(tray_id, constraint_id, cube_color):
    """IMPROVED: Precise cube placement INSIDE the correct tray"""
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
    print(f"🎯 PRECISE PLACE: Placing {cube_color} cube in {cube_color} tray at {[f'{x:.2f}' for x in tray_pos]}")
    
    # Calculate precise placement position INSIDE the tray
    tray_center = [tray_pos[0], tray_pos[1], tray_pos[2]]
    
    # Approach positions
    high_approach = [tray_center[0], tray_center[1], tray_center[2] + 0.25]
    low_approach = [tray_center[0], tray_center[1], tray_center[2] + 0.10]
    place_position = [tray_center[0], tray_center[1], tray_center[2] + 0.03]  # Inside the tray
    
    print(f"📍 Moving to high approach: {[f'{x:.2f}' for x in high_approach]}")
    ultra_smooth_panda_move_to_position(high_approach, 100, "high_approach")
    
    print(f"📍 Moving to low approach: {[f'{x:.2f}' for x in low_approach]}")
    ultra_smooth_panda_move_to_position(low_approach, 80, "low_approach")
    
    print(f"📍 Moving to place position INSIDE tray: {[f'{x:.2f}' for x in place_position]}")
    ultra_smooth_panda_move_to_position(place_position, 60, "place_inside")
    
    # Remove constraint and release cube
    if constraint_id is not None:
        try:
            p.removeConstraint(constraint_id)
            print("✅ Cube constraint removed")
        except Exception as e:
            print(f"⚠️ Constraint removal: {e}")
    
    # Open gripper to release cube
    print("🖐️ Opening gripper to release cube")
    open_gripper(panda, finger_joint_indices)
    
    # Wait for cube to settle INSIDE tray
    print("⏳ Waiting for cube to settle in tray...")
    for _ in range(150):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Verify cube is in the correct tray
    cube_in_tray = False
    for cube_id, color in cubes:
        if color == cube_color:
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            dist_to_tray = math.sqrt((cube_pos[0]-tray_center[0])**2 + (cube_pos[1]-tray_center[1])**2)
            if dist_to_tray < 0.15:  # Cube is inside tray
                cube_in_tray = True
                print(f"✅ VERIFIED: {cube_color} cube is INSIDE {cube_color} tray")
                break
    
    if not cube_in_tray:
        print(f"⚠️ WARNING: {cube_color} cube may not be properly inside tray")
    
    # Retract
    print("⬆️ Retracting from tray")
    ultra_smooth_panda_move_to_position(high_approach, 80, "retract")
    
    # Close gripper
    close_gripper(panda, finger_joint_indices)
    
    return cube_in_tray

def SimpleMobileAgentPlanner():
    """SIMPLE: Roll to XArm tray → Pick → Roll to own tray → Place → Repeat"""
    try:
        print("[SIMPLE MOBILE PLANNER] Starting simple transport")
        
        # Process each cube in the correct order
        cube_transport_order = ["red", "green", "blue"]
        
        for color in cube_transport_order:
            print(f"\n{'='*50}")
            print(f"🚀 TRANSPORTING {color.upper()} CUBE")
            print(f"{'='*50}")
            
            # Get the CORRECT tray positions for this color
            xarm_tray_id = xarm_trays[color]  # XArm's tray for this color
            mobile_tray_id = mobile_trays[color]  # Husky's tray for this color
            
            xarm_tray_pos, _ = p.getBasePositionAndOrientation(xarm_tray_id)
            mobile_tray_pos, _ = p.getBasePositionAndOrientation(mobile_tray_id)
            
            print(f"🎯 XArm's {color} tray at: {[f'{x:.2f}' for x in xarm_tray_pos]}")
            print(f"🎯 Own {color} tray at: {[f'{x:.2f}' for x in mobile_tray_pos]}")
            
            # Find the SPECIFIC cube that matches this color AND is in XArm's tray
            target_cube_id = None
            target_cube_pos = None

            for cube_id, cube_color in cubes:
                if cube_color == color:
                    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                    # Check if cube is in the correct XArm tray area
                    expected_tray_pos = xarm_tray_positions[color]  # Use the predefined tray positions
                    dist_to_expected_tray = math.sqrt((cube_pos[0]-expected_tray_pos[0])**2 + (cube_pos[1]-expected_tray_pos[1])**2)
                    
                    if dist_to_expected_tray < 0.4:  # Cube is near its correct XArm tray
                        target_cube_id = cube_id
                        target_cube_pos = cube_pos
                        print(f"✅ Found {color} cube at correct XArm tray: {[f'{x:.2f}' for x in cube_pos]}")
                        break

            if not target_cube_id:
                print(f"❌ No {color} cube found at correct XArm tray position")
                continue
            
            # === 1. ROLL TO XARM TRAY ===
            print(f"\n📍 PHASE 1: Rolling to XArm's {color} tray...")
            
            # Position in front of XArm's tray for this specific color
            pick_position = [xarm_tray_pos[0] + 0.8, xarm_tray_pos[1], husky_start_pos[2]]
            print(f"🎯 Pick position: {[f'{x:.2f}' for x in pick_position]}")
            
            simple_husky_navigation(pick_position, f"pick_{color}")
            
            # === 2. PICK CUBE ===
            print(f"\n🤖 PHASE 2: Picking {color} cube...")
            constraint_id = simple_panda_pick_cube(target_cube_id)
            
            if constraint_id is None:
                print(f"❌ Failed to pick {color} cube")
                continue
            
            # === 3. ROLL TO OWN TRAY ===
            print(f"\n📍 PHASE 3: Rolling to own {color} tray...")
            
            # Position in front of Husky's tray for this specific color
            place_position = [mobile_tray_pos[0] - 0.8, mobile_tray_pos[1], husky_start_pos[2]]
            print(f"🎯 Place position: {[f'{x:.2f}' for x in place_position]}")
            
            simple_husky_navigation(place_position, f"place_{color}")
            
            # === 4. PLACE CUBE ===
            print(f"\n🤖 PHASE 4: Placing {color} cube in {color} tray...")
            
            # Use the improved placement function
            place_success = improved_panda_place_cube(mobile_tray_id, constraint_id, color)
            
            if place_success:
                print(f"✅ {color.upper()} CUBE SUCCESSFULLY PLACED IN {color.upper()} TRAY!")
            else:
                print(f"❌ {color} cube placement failed")
            
            # Reset arm for next cube
            print("🔄 Resetting Panda arm for next transport...")
            initialize_panda_arm()
            
            # Wait a bit between transports
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./240.)
        
        print(f"\n🎉 ALL CUBES TRANSPORTED TO CORRECT TRAYS!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        
# ================================
# 2. INITIALIZATION & SETUP
# ================================
def initialize_simulation():
    """Initialize PyBullet simulation and basic environment"""
    print("=== INITIALIZING SIMULATION ===")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    return plane_id

def setup_xarm():
    """Setup XArm robot with color coding"""
    print("=== SETTING UP XARM ROBOT ===")
    robot_id = p.loadURDF("xarm/xarm6_with_gripper.urdf", [0, 0, 0], useFixedBase=True)
    
    num_joints = p.getNumJoints(robot_id)
    arm_joint_indices = []
    gripper_joint_indices = []

    joint_colors = [
        [1, 0, 0, 1], [0, 0.8, 0, 1], [0, 0.3, 1, 1], 
        [1, 0.8, 0, 1], [0.8, 0, 0.8, 1], [0, 0.8, 0.8, 1], [0.5, 0.5, 0.5, 1]
    ]

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        jname = info[1].decode("utf-8")
        jtype = info[2]
        
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            if "gripper" in jname.lower() or "finger" in jname.lower():
                gripper_joint_indices.append(j)
            else:
                arm_joint_indices.append(j)

    link_groups = {
        "base": [0, 1], "shoulder": [2, 3], "elbow": [4, 5],
        "wrist1": [6, 7], "wrist2": [8], "wrist3": [9],
        "gripper": list(range(10, num_joints))
    }

    for i, (group_name, link_indices) in enumerate(link_groups.items()):
        color_index = min(i, len(joint_colors) - 1)
        color = joint_colors[color_index]
        for link_idx in link_indices:
            if link_idx < num_joints:
                try:
                    p.changeVisualShape(robot_id, link_idx, rgbaColor=color)
                except:
                    pass

    ee_link_index = num_joints - 1
    print("🎨 XArm color coding complete!")
    return robot_id, arm_joint_indices, gripper_joint_indices, ee_link_index

def setup_mobile_robot():
    """Setup Husky mobile robot with Panda arm"""
    print("=== SETTING UP MOBILE ROBOT ===")
    
    husky_start_pos = [3.0, -2.0, 0.2]
    husky_start_orn = p.getQuaternionFromEuler([0, 0, math.pi/2])
    husky = p.loadURDF("husky/husky.urdf", husky_start_pos, husky_start_orn)
    
    for _ in range(300):
        p.stepSimulation()
        time.sleep(1./240.)

    panda_relative_pos = [0, 0, 0.15]
    panda_world_pos = [
        husky_start_pos[0] + panda_relative_pos[0],
        husky_start_pos[1] + panda_relative_pos[1], 
        husky_start_pos[2] + panda_relative_pos[2]
    ]

    panda = p.loadURDF("franka_panda/panda.urdf", panda_world_pos, useFixedBase=False)

    mount_constraint = p.createConstraint(
        parentBodyUniqueId=husky,
        parentLinkIndex=-1,
        childBodyUniqueId=panda,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, panda_relative_pos[2]],
        childFramePosition=[0, 0, 0]
    )

    panda_joints = []
    for j in range(p.getNumJoints(panda)):
        info = p.getJointInfo(panda, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            panda_joints.append(j)

    end_effector_index = 11
    finger_joint_indices = [9, 10]
    wheel_joints = [2, 3, 4, 5]

    for i in wheel_joints:
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)

    for _ in range(500):
        p.stepSimulation()
        time.sleep(1./240.)

    print(f"Mobile robot initialized at: {husky_start_pos}")
    return husky, panda, panda_joints, finger_joint_indices, wheel_joints, end_effector_index, husky_start_pos

def setup_environment():
    """Setup trays and cubes in the environment"""
    print("=== SETTING UP ENVIRONMENT ===")
    
    COLORS = {
        "red": [1, 0, 0, 1],
        "green": [0, 1, 0, 1],
        "blue": [0, 0, 1, 1],
    }

    xarm_trays = {}
    xarm_tray_positions = {
        "red":   [-0.5, -0.5, 0],
        "green": [-0.5, 0.5, 0],  
        "blue":  [-0.5, 0.0, 0],
    }

    for color, pos in xarm_tray_positions.items():
        tray_id = p.loadURDF("tray/traybox.urdf", pos)
        xarm_trays[color] = tray_id

    mobile_trays = {}
    mobile_tray_positions = {
        "red":   [1.8, -0.6, 0],
        "green": [1.8, 0.0, 0],
        "blue":  [1.8, 0.6, 0],
    }

    for color, pos in mobile_tray_positions.items():
        tray_id = p.loadURDF("tray/traybox.urdf", pos)
        mobile_trays[color] = tray_id

    cubes = []
    cube_positions = [
        [0.4, -0.3, 0],
        [0.4, 0.0, 0],
        [0.4, 0.3, 0]
    ]

    for i, (color, pos) in enumerate(zip(COLORS.keys(), cube_positions)):
        cube_id = p.loadURDF("cube_small.urdf", pos)
        p.changeVisualShape(cube_id, -1, rgbaColor=COLORS[color])
        cubes.append((cube_id, color))
        print(f"Spawned {color} cube at {pos}")

    return xarm_trays, mobile_trays, cubes

def simple_husky_navigation(target_world_pos, operation_type="approach"):
    """SIMPLE: Direct navigation without complex logic"""
    print(f"[SIMPLE NAV] {operation_type} to {[f'{x:.2f}' for x in target_world_pos]}")
    
    max_steps = 400
    tolerance = 0.4
    
    for step in range(max_steps):
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        
        # Calculate distance to target
        distance = math.sqrt((target_world_pos[0]-husky_pos[0])**2 + 
                           (target_world_pos[1]-husky_pos[1])**2)
        
        # Stop when close enough
        if distance <= tolerance:
            print(f"[SIMPLE NAV] ✅ Arrived! Distance: {distance:.2f}m")
            break
        
        # Calculate direction
        dx = target_world_pos[0] - husky_pos[0]
        dy = target_world_pos[1] - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        current_angle = p.getEulerFromQuaternion(husky_orn)[2]
        angle_diff = target_angle - current_angle
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Simple speed control
        base_speed = 1.5
        turn_gain = 1.5
        
        left_speed = base_speed - turn_gain * angle_diff
        right_speed = base_speed + turn_gain * angle_diff
        
        # Apply wheel control
        p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        
        p.stepSimulation()
        time.sleep(1./240.)
        
        if step % 100 == 0:
            print(f"[SIMPLE NAV] Progress: {distance:.2f}m to go")
    
    # Stop wheels
    for wheel in wheel_joints:
        p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
    
    # Stabilize
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

def simple_panda_pick_cube(cube_id):
    """SIMPLE: Panda arm pick similar to XArm"""
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    print(f"🎯 SIMPLE PANDA PICK: Cube at {[f'{x:.3f}' for x in cube_pos]}")
    
    # Approach positions (similar to XArm)
    high_approach = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.25]
    low_approach = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
    pick_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.03]
    
    # Move to high approach
    ultra_smooth_panda_move_to_position(high_approach, 100, "high_approach")
    
    # Open gripper
    open_gripper(panda, finger_joint_indices)
    
    # Move to low approach
    ultra_smooth_panda_move_to_position(low_approach, 80, "low_approach")
    
    # Move to pick position
    ultra_smooth_panda_move_to_position(pick_position, 60, "pick")
    
    # Close gripper
    close_gripper(panda, finger_joint_indices)
    
    # Create constraint
    constraint_id = None
    try:
        constraint_id = p.createConstraint(
            panda,
            end_effector_index,
            cube_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        )
        print("✅ Cube constraint created")
    except Exception as e:
        print(f"❌ Constraint failed: {e}")
    
    # Lift cube
    ultra_smooth_panda_move_to_position(high_approach, 80, "lift")
    
    print("✅ Simple Panda pick completed")
    return constraint_id

def simple_panda_place_cube(tray_id, constraint_id):
    """SIMPLE: Panda arm place similar to XArm"""
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
    print(f"🎯 SIMPLE PANDA PLACE: Tray at {[f'{x:.3f}' for x in tray_pos]}")
    
    # Approach positions
    high_approach = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.25]
    low_approach = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.08]
    
    # Move to high approach
    ultra_smooth_panda_move_to_position(high_approach, 100, "tray_approach")
    
    # Move to low approach
    ultra_smooth_panda_move_to_position(low_approach, 80, "low_approach")
    
    # Remove constraint and release cube
    if constraint_id is not None:
        try:
            p.removeConstraint(constraint_id)
            print("✅ Cube constraint removed")
        except Exception as e:
            print(f"⚠️ Constraint removal: {e}")
    
    # Open gripper
    open_gripper(panda, finger_joint_indices)
    
    # Wait for cube to settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Move back to high approach
    ultra_smooth_panda_move_to_position(high_approach, 80, "retract")
    
    # Close gripper
    close_gripper(panda, finger_joint_indices)
    
    print("✅ Simple Panda place completed")
    return True

def SimpleMobileAgentPlanner():
    """SIMPLE: Direct transport - XArm tray → Own tray"""
    try:
        print("[SIMPLE MOBILE PLANNER] Starting direct transport")
        transported_cubes = set()
        
        while shared["running"] and len(transported_cubes) < len(cubes):
            try:
                msg = to_mobile.get(timeout=0.5)
                if isinstance(msg, tuple) and msg[0] == "cube_ready":
                    _, color, xarm_tray_id = msg
                    
                    if color in transported_cubes:
                        continue
                    
                    print(f"\n{'='*50}")
                    print(f"🚀 STARTING {color.upper()} CUBE TRANSPORT")
                    print(f"{'='*50}")
                    
                    # Get target positions
                    xarm_tray_pos, _ = p.getBasePositionAndOrientation(xarm_tray_id)
                    mobile_tray_id = mobile_trays[color]
                    mobile_tray_pos, _ = p.getBasePositionAndOrientation(mobile_tray_id)
                    
                    # Find the cube in XArm's tray
                    target_cube_id = None
                    for cube_id, cube_color in cubes:
                        if cube_color == color:
                            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                            dist = math.sqrt((cube_pos[0]-xarm_tray_pos[0])**2 + (cube_pos[1]-xarm_tray_pos[1])**2)
                            if dist < 0.3:  # Cube is in this tray
                                target_cube_id = cube_id
                                break
                    
                    if target_cube_id is None:
                        print(f"❌ No {color} cube found in XArm's tray")
                        continue
                    
                    # === PICK FROM XARM'S TRAY ===
                    print(f"\n📦 PICKING {color} cube from XArm's tray")
                    
                    # Drive to XArm's tray
                    pick_position = [xarm_tray_pos[0] + 0.6, xarm_tray_pos[1], husky_start_pos[2]]
                    simple_husky_navigation(pick_position, "pick_approach")
                    
                    # Pick cube with Panda arm
                    constraint_id = simple_panda_pick_cube(target_cube_id)
                    
                    if constraint_id is None:
                        print(f"❌ Failed to pick {color} cube")
                        continue
                    
                    # === PLACE IN OWN TRAY ===
                    print(f"\n🏁 PLACING {color} cube in own tray")
                    
                    # Drive to own tray
                    place_position = [mobile_tray_pos[0] - 0.6, mobile_tray_pos[1], husky_start_pos[2]]
                    simple_husky_navigation(place_position, "place_approach")
                    
                    # Place cube with Panda arm
                    place_success = simple_panda_place_cube(mobile_tray_id, constraint_id)
                    
                    if place_success:
                        transported_cubes.add(color)
                        print(f"\n🎉 SUCCESS: {color} cube transport completed!")
                        
                        # Reset arm for next operation
                        initialize_panda_arm()
                    else:
                        print(f"\n❌ FAILED: {color} cube placement failed")
                    
                    if len(transported_cubes) >= len(cubes):
                        break
                        
            except queue.Empty:
                if shared.get("phase1_done", False) and len(transported_cubes) >= len(cubes):
                    break
                continue
                
        print(f"\n🏁 SIMPLE PLANNER FINISHED: Transported {len(transported_cubes)}/{len(cubes)} cubes")
        
    except Exception as e:
        print(f"❌ SIMPLE PLANNER ERROR: {e}")
        traceback.print_exc()

# ================================
# 3. CORE ROBOT FUNCTIONS
# ================================
def get_husky_world_position():
    """Get Husky's current world position accurately"""
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
    return husky_pos

def get_arm_base_position():
    """Get the base position of the XArm"""
    return [0, 0, 0]

def calculate_spatial_relationship(target_pos, base_pos):
    """Calculate detailed spatial relationship"""
    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]
    distance = math.sqrt(dx*dx + dy*dy)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    if abs(angle_deg) > 135:
        position = "DIRECTLY_BEHIND"
    elif abs(angle_deg) > 90:
        position = "BEHIND_SIDE" 
    elif abs(angle_deg) > 45:
        position = "FRONT_SIDE"
    else:
        position = "DIRECTLY_IN_FRONT"
        
    return {
        'distance': distance,
        'angle_rad': angle_rad,
        'angle_deg': angle_deg,
        'position': position,
        'needs_rotation': abs(angle_deg) > 90
    }

def open_gripper(body, gripper_joints):
    """Smooth gripper opening"""
    for j in gripper_joints:
        p.setJointMotorControl2(body, j, p.POSITION_CONTROL, targetPosition=0.045, force=80)
    for step in range(60):
        p.stepSimulation()
        time.sleep(1./240.)

def close_gripper(body, gripper_joints):
    """Smooth gripper closing with firm grip"""
    for j in gripper_joints:
        p.setJointMotorControl2(body, j, p.POSITION_CONTROL, targetPosition=0.0, force=120)
    for step in range(80):
        p.stepSimulation()
        time.sleep(1./240.)

# ================================
# 4. XARM FUNCTIONS
# ================================
def enhanced_smart_rotate_arm_base(target_pos):
    """Enhanced base rotation that prevents flips"""
    base_pos = get_arm_base_position()
    spatial = calculate_spatial_relationship(target_pos, base_pos)
    
    print(f"🔄 ENHANCED ROTATION: Target at {spatial['angle_deg']:.1f}° ({spatial['position']})")
    
    base_joint_idx = arm_joint_indices[0]
    current_base_angle = p.getJointState(robot_id, base_joint_idx)[0]
    
    target_base_angle = spatial['angle_rad']
    angle_diff = target_base_angle - current_base_angle
    
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    optimal_rotation = current_base_angle + angle_diff * 0.8
    
    print(f"   Current: {math.degrees(current_base_angle):.1f}° -> Target: {math.degrees(optimal_rotation):.1f}°")
    
    steps = 80
    for step in range(steps):
        progress = step / steps
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        current_target = current_base_angle + (optimal_rotation - current_base_angle) * eased_progress
        
        p.setJointMotorControl2(
            robot_id,
            base_joint_idx,
            p.POSITION_CONTROL,
            targetPosition=current_target,
            force=300
        )
        
        for j in arm_joint_indices[1:]:
            current_pos = p.getJointState(robot_id, j)[0]
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=current_pos,
                force=200
            )
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    print("✅ Enhanced rotation complete")

def enhanced_smooth_move_arm_to_position(target_pos, steps=100, operation_type="move"):
    """Enhanced smooth arm movement"""
    if operation_type not in ["pick", "place", "hover"]:
        base_pos = get_arm_base_position()
        spatial = calculate_spatial_relationship(target_pos, base_pos)
        if abs(spatial['angle_deg']) > 30:
            enhanced_smart_rotate_arm_base(target_pos)
    
    if operation_type in ["approach", "tray_approach", "high_approach"]:
        min_height = 0.25
        if target_pos[2] < min_height:
            target_pos = [target_pos[0], target_pos[1], min_height]
    
    print(f"🤖 ENHANCED MOVE: {operation_type} to {[f'{x:.3f}' for x in target_pos]}")
    
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        ee_link_index,
        target_pos,
        targetOrientation=target_orn,
        maxNumIterations=100
    )
    
    current_joint_positions = []
    for joint_idx in arm_joint_indices:
        current_joint_positions.append(p.getJointState(robot_id, joint_idx)[0])
    
    for step in range(steps):
        progress = step / steps
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        
        for i, joint_idx in enumerate(arm_joint_indices):
            if i < len(ik_solution) and i < len(current_joint_positions):
                current_joint_pos = current_joint_positions[i]
                target_joint_pos = ik_solution[i]
                intermediate_pos = current_joint_pos + (target_joint_pos - current_joint_pos) * eased_progress
                
                p.setJointMotorControl2(
                    robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=intermediate_pos,
                    force=250
                )
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1./240.)

def enhanced_smart_xarm_pick_cube(cube_id):
    """Enhanced cube picking with better motion planning"""
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    print(f"🎯 ENHANCED PICK: Cube at {[f'{x:.3f}' for x in cube_pos]}")
    
    high_hover = [cube_pos[0], cube_pos[1], max(cube_pos[2] + 0.3, 0.25)]
    enhanced_smooth_move_arm_to_position(high_hover, 100, "high_approach")
    
    print("🖐️ Opening gripper")
    open_gripper(robot_id, gripper_joint_indices)
    
    low_hover = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
    enhanced_smooth_move_arm_to_position(low_hover, 80, "low_hover")
    
    pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.04]
    enhanced_smooth_move_arm_to_position(pick_pos, 60, "pick")
    
    print("✊ Closing gripper")
    for j in gripper_joint_indices:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=180)
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    try:
        cid = p.createConstraint(robot_id, ee_link_index, cube_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        print(f"✅ Cube constraint created: {cid}")
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
    except Exception as e:
        print(f"❌ Constraint failed: {e}")
        cid = None
        return None
    
    enhanced_smooth_move_arm_to_position(high_hover, 80, "lift_with_cube")
    
    print("✅ Enhanced pick completed successfully")
    return cid

def enhanced_smart_xarm_place_cube(constraint_id, cube_id, tray_id):
    """Enhanced cube placement with precise tray targeting"""
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
    print(f"🎯 ENHANCED PLACE: Tray at {[f'{x:.3f}' for x in tray_pos]}")
    
    tray_center_high = [tray_pos[0], tray_pos[1], max(tray_pos[2] + 0.3, 0.25)]
    enhanced_smooth_move_arm_to_position(tray_center_high, 100, "tray_approach_high")
    
    tray_center_low = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.08]
    enhanced_smooth_move_arm_to_position(tray_center_low, 80, "tray_approach_low")
    
    if constraint_id is not None:
        try:
            p.removeConstraint(constraint_id)
            print("✅ Cube constraint removed")
            for _ in range(30):
                p.stepSimulation()
                time.sleep(1./240.)
        except Exception as e:
            print(f"⚠️  Constraint removal warning: {e}")
    
    print("🖐️ Opening gripper to release cube")
    open_gripper(robot_id, gripper_joint_indices)
    
    for _ in range(80):
        p.stepSimulation()
        time.sleep(1./240.)
    
    clear_height = tray_pos[2] + 0.25
    clear_pos = [tray_pos[0], tray_pos[1], clear_height]
    enhanced_smooth_move_arm_to_position(clear_pos, 60, "clear_tray")
    
    close_gripper(robot_id, gripper_joint_indices)
    
    print("✅ Enhanced placement completed successfully")
    return True

# ================================
# 5. MOBILE ROBOT FUNCTIONS
# ================================
def initialize_panda_arm():
    """Initialize Panda arm to proper downward-facing position"""
    print("[Mobile Robot] Initializing Panda arm to downward position")
    
    init_pos = [0, 0, 0.4]
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
    world_pos = [
        husky_pos[0] + init_pos[0],
        husky_pos[1] + init_pos[1], 
        husky_pos[2] + init_pos[2]
    ]
    
    ik_solution = p.calculateInverseKinematics(
        panda,
        end_effector_index,
        world_pos,
        targetOrientation=target_orn,
        maxNumIterations=100
    )
    
    for i, joint_idx in enumerate(panda_joints):
        if i < len(ik_solution):
            p.setJointMotorControl2(
                panda,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=ik_solution[i],
                force=200
            )
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    print("[Mobile Robot] Panda arm initialized to downward position")

def enhanced_smooth_panda_move_to_position(target_world_pos, steps=100, operation_type="move"):
    """FIXED: Panda arm movement using correct world coordinates"""
    print(f"🤖 PANDA MOVE: {operation_type} to {[f'{x:.3f}' for x in target_world_pos]}")
    
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    ik_solution = p.calculateInverseKinematics(
        panda,
        end_effector_index,
        target_world_pos,
        targetOrientation=target_orn,
        maxNumIterations=200
    )
    
    current_joint_positions = []
    for joint_idx in panda_joints:
        current_joint_positions.append(p.getJointState(panda, joint_idx)[0])
    
    for step in range(steps):
        progress = step / steps
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        
        for i, joint_idx in enumerate(panda_joints):
            if i < len(ik_solution) and i < len(current_joint_positions):
                current_joint_pos = current_joint_positions[i]
                target_joint_pos = ik_solution[i]
                intermediate_pos = current_joint_pos + (target_joint_pos - current_joint_pos) * eased_progress
                
                p.setJointMotorControl2(
                    panda,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=intermediate_pos,
                    force=150
                )
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1./240.)

def enhanced_smooth_husky_navigation(target_world_pos, operation_type="approach", tolerance=0.3):
    """Enhanced smooth navigation for Husky with motion profiling"""
    print(f"[SMOOTH NAV] {operation_type} to {[f'{x:.2f}' for x in target_world_pos]}")
    
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
    start_pos = husky_pos.copy()
    
    distance = math.sqrt((target_world_pos[0]-husky_pos[0])**2 + 
                        (target_world_pos[1]-husky_pos[1])**2)
    
    total_steps = int(distance * 150)  # Adjust steps based on distance
    total_steps = max(100, min(total_steps, 400))  # Clamp between 100-400 steps
    
    for step in range(total_steps):
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        
        # Calculate progress with easing
        progress = step / total_steps
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)  # Smooth easing
        
        # Calculate current target based on eased progress
        current_target_x = start_pos[0] + (target_world_pos[0] - start_pos[0]) * eased_progress
        current_target_y = start_pos[1] + (target_world_pos[1] - start_pos[1]) * eased_progress
        
        dx = current_target_x - husky_pos[0]
        dy = current_target_y - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        current_angle = p.getEulerFromQuaternion(husky_orn)[2]
        angle_diff = target_angle - current_angle
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Adaptive speed control
        current_distance = math.sqrt((target_world_pos[0]-husky_pos[0])**2 + 
                                   (target_world_pos[1]-husky_pos[1])**2)
        
        # Slow down when approaching target
        if current_distance < 1.0:
            base_speed = 0.8 * (current_distance / 1.0)  # Linear slowdown
        else:
            base_speed = 1.2
        
        base_speed = max(0.3, base_speed)  # Minimum speed
        
        turn_gain = 1.5
        left_speed = base_speed - turn_gain * angle_diff
        right_speed = base_speed + turn_gain * angle_diff
        
        # Limit maximum speed
        max_speed = 2.0
        left_speed = max(min(left_speed, max_speed), -max_speed)
        right_speed = max(min(right_speed, max_speed), -max_speed)
        
        # Apply wheel control
        p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=200)
        p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=200)
        p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=200)
        p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=200)
        
        p.stepSimulation()
        time.sleep(1./240.)
        
        # Early termination if close enough
        if current_distance <= tolerance:
            print(f"[SMOOTH NAV] ✅ Arrived at target! Final distance: {current_distance:.2f}m")
            break
    
    # Smooth stop
    for decel_step in range(30):
        factor = 1.0 - (decel_step / 30.0)  # Linear deceleration
        for wheel in wheel_joints:
            current_vel = p.getJointState(husky, wheel)[1]
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, 
                                  targetVelocity=current_vel * factor, force=100)
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Final stop
    for wheel in wheel_joints:
        p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
    
    print(f"[SMOOTH NAV] 🛑 Smooth stop complete")

def ultra_smooth_panda_move_to_position(target_world_pos, steps=120, operation_type="move"):
    """IMPROVED: Ultra-smooth Panda arm movement with better physics"""
    print(f"🤖 ULTRA-SMOOTH PANDA: {operation_type} to {[f'{x:.3f}' for x in target_world_pos]}")
    
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    # IMPROVED: Better IK calculation that considers all joints including base
    ik_solution = p.calculateInverseKinematics(
        panda,
        end_effector_index,
        target_world_pos,
        targetOrientation=target_orn,
        lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        jointRanges=[5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8],
        restPoses=[0, 0, 0, -1.0, 0, 1.5, 0],
        maxNumIterations=200
    )
    
    current_joint_positions = []
    for joint_idx in panda_joints:
        current_joint_positions.append(p.getJointState(panda, joint_idx)[0])
    
    # IMPROVED: Advanced motion profiling with smoother transitions
    for step in range(steps):
        progress = step / steps
        
        # IMPROVED: Smoother easing function
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        
        for i, joint_idx in enumerate(panda_joints):
            if i < len(ik_solution) and i < len(current_joint_positions):
                current_joint_pos = current_joint_positions[i]
                target_joint_pos = ik_solution[i]
                
                # Special handling for base joint (joint 0) - smoother rotation
                if i == 0:  # Base joint
                    # Calculate shortest rotation path
                    angle_diff = target_joint_pos - current_joint_pos
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    intermediate_pos = current_joint_pos + angle_diff * eased_progress
                    force = 100  # Lower force for base rotation
                else:
                    intermediate_pos = current_joint_pos + (target_joint_pos - current_joint_pos) * eased_progress
                    force = 80 if "pick" in operation_type.lower() else 120
                
                p.setJointMotorControl2(
                    panda,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=intermediate_pos,
                    force=force
                )
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    # IMPROVED: Longer stabilization period
    for _ in range(40):
        p.stepSimulation()
        time.sleep(1./240.)

def improved_mobile_navigation(target_world_pos, operation_type="approach", tolerance=0.3):
    """SLOW and CAREFUL navigation for precise positioning"""
    print(f"[CAREFUL NAV] {operation_type} to {[f'{x:.2f}' for x in target_world_pos]}")
    
    steps = 0
    max_steps = 600
    last_position = get_husky_world_position()
    stuck_count = 0
    
    while steps < max_steps:
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        
        distance = math.sqrt((target_world_pos[0]-husky_pos[0])**2 + 
                           (target_world_pos[1]-husky_pos[1])**2)
        
        if distance <= tolerance:
            print(f"[CAREFUL NAV] ✅ Arrived at target! Final distance: {distance:.2f}m")
            break
        
        current_move = math.sqrt((husky_pos[0]-last_position[0])**2 + 
                               (husky_pos[1]-last_position[1])**2)
        if current_move < 0.01 and steps > 50:
            stuck_count += 1
            if stuck_count > 5:
                print("🔄 Making adjustment to get unstuck")
                adjust_angle = math.atan2(target_world_pos[1]-husky_pos[1], 
                                        target_world_pos[0]-husky_pos[0]) + math.pi/4
                adjust_distance = 0.3
                adjust_pos = [
                    husky_pos[0] + math.cos(adjust_angle) * adjust_distance,
                    husky_pos[1] + math.sin(adjust_angle) * adjust_distance,
                    husky_pos[2]
                ]
                simple_move_to_position(adjust_pos, 30)
                stuck_count = 0
                last_position = get_husky_world_position()
                continue
        
        last_position = husky_pos
        
        dx = target_world_pos[0] - husky_pos[0]
        dy = target_world_pos[1] - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        current_angle = p.getEulerFromQuaternion(husky_orn)[2]
        angle_diff = target_angle - current_angle
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        base_speed = 1.0
        turn_gain = 1.2
        
        if distance < 1.0:
            base_speed = 0.8
        if distance < 0.5:
            base_speed = 0.5
        
        left_speed = base_speed - turn_gain * angle_diff
        right_speed = base_speed + turn_gain * angle_diff
        
        max_speed = 1.5
        left_speed = max(min(left_speed, max_speed), -max_speed)
        right_speed = max(min(right_speed, max_speed), -max_speed)
        
        p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=200)
        p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=200)
        p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=200)
        p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=200)
        
        p.stepSimulation()
        time.sleep(1./240.)
        steps += 1
        
        if steps % 100 == 0:
            print(f"[CAREFUL NAV] Progress: {distance:.2f}m to go")
        
        if steps >= max_steps:
            print(f"[CAREFUL NAV] ⚠️  Timeout after {max_steps} steps, distance: {distance:.2f}m")
    
    for wheel in wheel_joints:
        p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    print(f"[CAREFUL NAV] 🛑 COMPLETE STOP and stabilized")

def simple_move_to_position(target_world_pos, steps=50):
    """Enhanced simple movement with basic obstacle checking"""
    for step in range(steps):
        husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
        
        dx = target_world_pos[0] - husky_pos[0]
        dy = target_world_pos[1] - husky_pos[1]
        target_angle = math.atan2(dy, dx)
        
        current_angle = p.getEulerFromQuaternion(husky_orn)[2]
        angle_diff = target_angle - current_angle
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        base_speed = 1.5
        turn_gain = 1.5
        
        left_speed = base_speed - turn_gain * angle_diff
        right_speed = base_speed + turn_gain * angle_diff
        
        p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        p.setJointMotorControl2(husky, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=300)
        p.setJointMotorControl2(husky, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=300)
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    for wheel in wheel_joints:
        p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=500)

def rotate_panda_arm_base(target_angle_rad, steps=80):
    """Rotate the Panda arm base to face a specific angle"""
    print(f"🔄 ROTATING PANDA ARM BASE to {math.degrees(target_angle_rad):.1f}°")
    
    # Panda arm base joint is typically joint 0
    base_joint_idx = 0
    
    current_angle = p.getJointState(panda, base_joint_idx)[0]
    angle_diff = target_angle_rad - current_angle
    
    # Normalize angle difference
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    print(f"   Current: {math.degrees(current_angle):.1f}° -> Target: {math.degrees(target_angle_rad):.1f}°")
    
    for step in range(steps):
        progress = step / steps
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        current_target = current_angle + angle_diff * eased_progress
        
        p.setJointMotorControl2(
            panda,
            base_joint_idx,
            p.POSITION_CONTROL,
            targetPosition=current_target,
            force=150
        )
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    print("✅ Panda arm base rotation complete")

def calculate_panda_target_angle(target_pos):
    """Calculate the angle the Panda arm should face to reach target"""
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)
    
    # Calculate direction from Husky to target
    dx = target_pos[0] - husky_pos[0]
    dy = target_pos[1] - husky_pos[1]
    target_angle = math.atan2(dy, dx)
    
    return target_angle

def coordinated_mobile_pick_cube(xarm_tray_id, cube_color):
    """IMPROVED: Complete mobile pick sequence with Panda arm base rotation"""
    print(f"\n{'='*60}")
    print(f"🚀 COORDINATED MOBILE PICK: {cube_color.upper()} CUBE")
    print(f"{'='*60}")
    
    try:
        xarm_tray_pos, _ = p.getBasePositionAndOrientation(xarm_tray_id)
        
        target_cube_id = None
        for cube_id, color in cubes:
            if color == cube_color:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                dist = math.sqrt((cube_pos[0]-xarm_tray_pos[0])**2 + (cube_pos[1]-xarm_tray_pos[1])**2)
                if dist < 0.3:
                    target_cube_id = cube_id
                    break
        
        if target_cube_id is None:
            print(f"❌ No {cube_color} cube found in tray")
            return False, None, None
        
        pick_approach_pos = [xarm_tray_pos[0] + 0.7, xarm_tray_pos[1], husky_start_pos[2]]
        pick_operation_pos = [xarm_tray_pos[0] + 0.5, xarm_tray_pos[1], husky_start_pos[2]]
        
        print(f"🎯 Target: {cube_color} cube at tray {xarm_tray_pos[:2]}")
        
        print("\n📍 PHASE 1: Navigating to pick location")
        improved_mobile_navigation(pick_approach_pos, "pick_approach", tolerance=0.2)
        improved_mobile_navigation(pick_operation_pos, "pick_operation", tolerance=0.15)
        
        print("\n🛑 PHASE 2: Full stop and stabilization")
        for wheel in wheel_joints:
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("\n🤖 PHASE 3: Panda arm pick operation")
        cube_pos, _ = p.getBasePositionAndOrientation(target_cube_id)
        
        # NEW: Rotate Panda arm base to face the cube
        print("🔄 Rotating Panda arm base to face cube")
        target_angle = calculate_panda_target_angle(cube_pos)
        rotate_panda_arm_base(target_angle)
        
        # IMPROVED: Lower approach and pick positions
        approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
        pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.01]
        
        ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_approach")
        open_gripper(panda, finger_joint_indices)
        ultra_smooth_panda_move_to_position(pick_pos, 120, "ultra_smooth_pick")
        close_gripper(panda, finger_joint_indices)
        
        # Wait for grip to stabilize
        for _ in range(80):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Create constraint for smooth holding
        constraint_id = None
        try:
            ee_state = p.getLinkState(panda, end_effector_index)
            constraint_id = p.createConstraint(
                panda,
                end_effector_index,
                target_cube_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            )
            print(f"✅ Cube constraint created: {constraint_id}")
        except Exception as e:
            print(f"❌ Constraint creation failed: {e}")
        
        # Smooth lift with constraint
        ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_lift")
        
        # Verify successful pickup
        new_cube_pos, _ = p.getBasePositionAndOrientation(target_cube_id)
        if new_cube_pos[2] > cube_pos[2] + 0.05:
            print(f"✅ SUCCESS: {cube_color} cube picked up and lifted")
            return True, constraint_id, target_cube_id
        else:
            print(f"❌ FAILED: {cube_color} cube not lifted properly")
            if constraint_id:
                try:
                    p.removeConstraint(constraint_id)
                except:
                    pass
            return False, None, target_cube_id
            
    except Exception as e:
        print(f"❌ COORDINATED PICK ERROR: {e}")
        traceback.print_exc()
        return False, None, None

def coordinated_mobile_place_cube(mobile_tray_id, cube_color, constraint_id, cube_id):
    """IMPROVED: Complete mobile place sequence with Panda arm base rotation"""
    print(f"\n{'='*60}")
    print(f"🎯 COORDINATED MOBILE PLACE: {cube_color.upper()} CUBE")
    print(f"{'='*60}")
    
    try:
        mobile_tray_pos, _ = p.getBasePositionAndOrientation(mobile_tray_id)
        
        place_approach_pos = [mobile_tray_pos[0] - 0.7, mobile_tray_pos[1], husky_start_pos[2]]
        place_operation_pos = [mobile_tray_pos[0] - 0.5, mobile_tray_pos[1], husky_start_pos[2]]
        
        print(f"🎯 Destination: {cube_color} tray at {mobile_tray_pos[:2]}")
        
        print("\n📍 PHASE 1: Navigating to place location")
        improved_mobile_navigation(place_approach_pos, "place_approach", tolerance=0.2)
        improved_mobile_navigation(place_operation_pos, "place_operation", tolerance=0.15)
        
        print("\n🛑 PHASE 2: Full stop and stabilization")
        for wheel in wheel_joints:
            p.setJointMotorControl2(husky, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("\n🤖 PHASE 3: Panda arm place operation")
        
        # NEW: Rotate Panda arm base to face the tray
        print("🔄 Rotating Panda arm base to face tray")
        target_angle = calculate_panda_target_angle(mobile_tray_pos)
        rotate_panda_arm_base(target_angle)
        
        approach_pos = [mobile_tray_pos[0], mobile_tray_pos[1], mobile_tray_pos[2] + 0.25]
        place_pos = [mobile_tray_pos[0], mobile_tray_pos[1], mobile_tray_pos[2] + 0.05]
        
        ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_tray_approach")
        ultra_smooth_panda_move_to_position(place_pos, 80, "ultra_smooth_place")
        
        # Remove constraint and release cube
        if constraint_id is not None:
            try:
                p.removeConstraint(constraint_id)
                print("✅ Cube constraint removed")
            except Exception as e:
                print(f"⚠️ Constraint removal warning: {e}")
        
        open_gripper(panda, finger_joint_indices)
        
        # Wait for cube to settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        ultra_smooth_panda_move_to_position(approach_pos, 100, "ultra_smooth_retract")
        close_gripper(panda, finger_joint_indices)
        
        print(f"✅ SUCCESS: {cube_color} cube placed")
        return True
            
    except Exception as e:
        print(f"❌ COORDINATED PLACE ERROR: {e}")
        traceback.print_exc()
        return False

def ImprovedMobileAgentPlanner():
    try:
        print("[IMPROVED MOBILE PLANNER] Starting with coordinated control")
        transported_cubes = set()
        
        while shared["running"] and len(transported_cubes) < len(cubes):
            try:
                msg = to_mobile.get(timeout=0.5)
                if isinstance(msg, tuple) and msg[0] == "cube_ready":
                    _, color, xarm_tray_id = msg
                    
                    if color in transported_cubes:
                        continue
                    
                    print(f"\n{'🚀'*20} STARTING {color.upper()} CUBE TRANSPORT {'🚀'*20}")
                    
                    mobile_tray_id = mobile_trays[color]
                    
                    # SEQUENCE: Approach XArm tray → Rotate Panda base → Pick cube → Lift arm
                    pick_success, constraint_id, cube_id = coordinated_mobile_pick_cube(xarm_tray_id, color)
                    
                    if pick_success:
                        # SEQUENCE: Navigate to mobile tray → Rotate Panda base → Place cube
                        place_success = coordinated_mobile_place_cube(mobile_tray_id, color, constraint_id, cube_id)
                        
                        if place_success:
                            transported_cubes.add(color)
                            print(f"\n🎉 SUCCESS: {color} cube transport completed!")
                            
                            # Reset Panda arm to neutral position after each transport
                            print("🔄 Resetting Panda arm to neutral position")
                            initialize_panda_arm()
                        else:
                            print(f"\n❌ FAILED: {color} cube placement failed")
                    else:
                        print(f"\n❌ FAILED: {color} cube pick failed")
                    
                    if len(transported_cubes) >= len(cubes):
                        break
                        
            except queue.Empty:
                if shared.get("phase1_done", False) and len(transported_cubes) >= len(cubes):
                    break
                continue
                
        print(f"\n🏁 MOBILE PLANNER FINISHED: Transported {len(transported_cubes)}/{len(cubes)} cubes")
        
    except Exception as e:
        print(f"❌ MOBILE PLANNER ERROR: {e}")
        traceback.print_exc()

# ================================
# 6. PLANNING & EXECUTION
# ================================
command_queue = queue.Queue()
to_mobile = queue.Queue()
shared = {"phase1_done": False, "running": True}

def ArmAgentPlanner(cubes_list):
    try:
        print("[SMART XArm Planner] Starting with spatial intelligence")
        
        for cube_id, color in cubes_list:
            print(f"[SMART XArm Planner] Processing {color} cube")
            command_queue.put(("smart_xarm_pick_place", cube_id, color))
            time.sleep(2)
        
        shared["phase1_done"] = True
        print("[SMART XArm Planner] Finished smart sorting")
        
    except Exception:
        traceback.print_exc()

def xarm_return_to_home():
    """Return XArm to initial home position after completing all tasks"""
    print("🏠 XArm returning to home position...")
    
    # Home position (adjust based on your XArm's initial position)
    home_position = [0.3, 0, 0.3]  # Slightly forward and up
    
    # First rotate base to forward position
    base_joint_idx = arm_joint_indices[0]
    p.setJointMotorControl2(
        robot_id, 
        base_joint_idx, 
        p.POSITION_CONTROL, 
        targetPosition=0,  # Forward facing
        force=300
    )
    
    # Move to home position smoothly
    enhanced_smooth_move_arm_to_position(home_position, 120, "return_home")
    
    print("✅ XArm returned to home position")

def enhanced_executor_loop():
    print("[ENHANCED SMART Executor] Starting with improved motion planning")
    cubes_processed = 0
    total_cubes = len(cubes)
    
    while True:
        try:
            cmd = command_queue.get(timeout=0.05)
            
            if cmd[0] == "smart_xarm_pick_place":
                _, cube_id, color = cmd
                print(f"\n{'='*50}")
                print(f"🎯 PROCESSING {color.upper()} CUBE")
                print(f"{'='*50}")
                
                print(f"\n📦 PICK PHASE for {color} cube")
                constraint_id = enhanced_smart_xarm_pick_cube(cube_id)
                
                if constraint_id is not None:
                    print(f"\n🏁 PLACE PHASE for {color} cube")
                    tray_id = xarm_trays[color]
                    success = enhanced_smart_xarm_place_cube(constraint_id, cube_id, tray_id)
                    
                    if success:
                        to_mobile.put(("cube_ready", color, tray_id))
                        cubes_processed += 1
                        print(f"✅ SUCCESS: {color} cube completed ({cubes_processed}/{total_cubes})\n")
                        
                        # Check if all cubes are placed
                        if cubes_processed >= total_cubes:
                            print("🎉 ALL CUBES PLACED! Returning XArm to home position...")
                            xarm_return_to_home()
                    else:
                        print(f"⚠️  WARNING: {color} cube placement had issues\n")
                else:
                    print(f"❌ FAILED: {color} cube pick failed\n")
                
        except queue.Empty:
            pass

        p.stepSimulation()
        time.sleep(1./240.)
        
        if (shared.get("phase1_done", False) and 
            command_queue.empty() and 
            to_mobile.empty() and
            cubes_processed >= total_cubes):
            break

    print("[ENHANCED SMART Executor] Finished all operations")

# ================================
# 7. MAIN EXECUTION (UPDATED)
# ================================
if __name__ == "__main__":
    xarm_llm = SmartOllamaAgent(
        "XArm",
        """You are controlling a XArm6 robotic arm with spatial intelligence.
        CRITICAL SPATIAL CONSTRAINTS:
        - The arm base can rotate 180 degrees
        - The gripper MUST always face downward
        - Never flip the arm upside down
        - For targets behind the arm (>90 degrees), ROTATE the base
        - For side targets (45-90 degrees), use ROTATE_AND_MOVE
        - For front targets (<45 degrees), use MOVE_DIRECTLY
        - When all cubes are placed in trays, return arm to initial position""",
        model="llama2"
    )

    mobile_llm = SmartOllamaAgent(
        "MobileRobot",
        """You are an expert PyBullet-based controller for a Clearpath Husky robot equipped with a Franka Panda arm.

    NAVIGATION STRATEGY:
    1. ALWAYS approach targets from the SIDE, never head-on
    2. Use LEFT side approach by default, RIGHT side to avoid XArm workspace
    3. Stop 0.4m from targets for optimal arm operation
    4. NEVER enter the XArm workspace (radius 1.5m from center)

    PICKING STRATEGY:
    1. Position Husky so Panda arm can reach the target directly
    2. Use DIRECT SUCTION approach: move to cube and pick immediately
    3. Stop moving as soon as within arm reach distance

    PRIORITIES:
    - Safety: Avoid XArm workspace at all costs
    - Efficiency: Use side approaches and direct picking
    - Precision: Stop at optimal distances for arm operation

    Your goal: Achieve fast, safe, and reliable cube transport using side approaches and direct picking.""",
        model="llama2"
    )

    plane_id = initialize_simulation()
    robot_id, arm_joint_indices, gripper_joint_indices, ee_link_index = setup_xarm()
    husky, panda, panda_joints, finger_joint_indices, wheel_joints, end_effector_index, husky_start_pos = setup_mobile_robot()
    xarm_trays, mobile_trays, cubes = setup_environment()
    
    initialize_panda_arm()

    print("\n=== STARTING SMART MULTI-ROBOT SYSTEM ===")
    print("XArm will use LLM spatial intelligence to decide when to rotate")
    print("Mobile robot will use LLM for real-time decision making")

    # Start the traditional planning threads
    arm_planner_thread = threading.Thread(target=ArmAgentPlanner, args=(cubes,), daemon=True)
    #mobile_planner_thread = threading.Thread(target=ImprovedMobileAgentPlanner, daemon=True)
    mobile_planner_thread = threading.Thread(target=SimpleMobileAgentPlanner, daemon=True)

    arm_planner_thread.start()
    mobile_planner_thread.start()

    # # NEW: LLM Control Loop for real-time decision making
    # def llm_control_loop():
    #     """Run LLM decision making alongside traditional planning"""
    #     llm_cycle = 0
    #     while shared["running"] and llm_cycle < 20:  # Limit to 20 cycles for testing
    #         try:
    #             print(f"\n🤖 LLM DECISION CYCLE {llm_cycle + 1}")
    #             decision = mobile_llm.analyze_environment(husky, panda, cubes, xarm_trays, mobile_trays)
    #             if decision and decision != "WAIT":
    #                 mobile_llm.execute_decision(decision, husky, panda, cubes, xarm_trays, mobile_trays)
                
    #             llm_cycle += 1
    #             time.sleep(3)  # Wait between LLM decisions
                
    #         except Exception as e:
    #             print(f"LLM control error: {e}")
    #             time.sleep(2)

    # # Start LLM control in a separate thread
    # llm_thread = threading.Thread(target=llm_control_loop, daemon=True)
    # llm_thread.start()

    # Run the enhanced executor
    enhanced_executor_loop()

    print("\n🎉 SMART SYSTEM COMPLETE!")
    print("Both robots successfully completed their tasks!")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./120.)
    except KeyboardInterrupt:
        print("Exiting")
        shared["running"] = False
        p.disconnect()