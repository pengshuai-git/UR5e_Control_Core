import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import serial

# 导入 IK 和 FK 两个核心模块
from Core_IK import TrajectoryFollower
from Core_FK import ForwardKinematicsController

# ================= 控制参数 =================
MAX_JOYSTICK_SPEED = 0.3    # 摇杆满舵时的最大移动速度 (m/s)
MAX_POS = [0.8, 0.6, 0.6] 
MIN_POS = [-0.3, -0.6, 0.1]

IK_GAIN_POS = 40.0
IK_GAIN_ROT = 20.0
MAX_JOINT_VEL = 8.0

Q_INIT = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# ================= 全局变量 =================
target_pos = None
target_quat = None
target_joints = Q_INIT.copy()  # 【新增】存放 6 个关节的目标角度

running = True
need_reset = False
control_mode = "IK"            # 【新增】控制模式："IK" 或 "FK"

joystick_vel = np.zeros(3)

# ================= 串口通信线程 =================
def serial_listener_loop(port="COM4", baudrate=115200):
    global joystick_vel, running, need_reset, control_mode, target_joints
    try:
        ser = serial.Serial(port, baudrate, timeout=0.01)
        ser.setDTR(True)
        ser.setRTS(True)
        print(f"✅ 成功连接到串口 {port}，等待 VOFA+ 数据...")
        
        while running:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    if ':' in line:
                        prefix, vals_str = line.split(':', 1)
                        vals = [float(v) for v in vals_str.split(',')]
                        
                        # 1. 监听复位指令 "p:0,0,0"
                        if prefix == 'p':
                            need_reset = True
                            continue

                        # 2. 解析 FK 滑条指令 (a1 到 a6)
                        if prefix in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'] and len(vals) >= 1:
                            control_mode = "FK"  # 自动切入正运动学模式
                            joint_idx = int(prefix[1]) - 1
                            
                            # 将 VOFA+ 的 [0, 1000] 映射到 [-3.14, 3.14] 弧度 (-180° 到 180°)
                            # 500 对应 0 弧度
                            raw_val = vals[0]
                            angle_rad = ((raw_val - 500.0) / 500.0) * np.pi
                            target_joints[joint_idx] = angle_rad
                            continue

                        # 3. 摇杆死区处理函数
                        def apply_deadzone(v):
                            norm = v / 1000.0
                            return 0.0 if abs(norm) < 0.05 else norm

                        # 4. 解析 IK 摇杆/滑块指令 (切换回 IK 模式)
                        if prefix in ['x', 'y', 'z', 'xy', 'xz', 'yz']:
                            control_mode = "IK"  # 自动切入逆运动学模式

                            if prefix == 'xy' and len(vals) >= 2:
                                joystick_vel[0] = apply_deadzone(vals[0])
                                joystick_vel[1] = apply_deadzone(vals[1])
                                joystick_vel[2] = 0.0
                            elif prefix == 'xz' and len(vals) >= 2:
                                joystick_vel[0] = apply_deadzone(vals[0])
                                joystick_vel[2] = apply_deadzone(vals[1])
                                joystick_vel[1] = 0.0
                            elif prefix == 'yz' and len(vals) >= 2:
                                joystick_vel[1] = apply_deadzone(vals[0])
                                joystick_vel[2] = apply_deadzone(vals[1])
                                joystick_vel[0] = 0.0
                            elif prefix == 'x' and len(vals) >= 1:
                                joystick_vel[0] = apply_deadzone(vals[0])
                            elif prefix == 'y' and len(vals) >= 1:
                                joystick_vel[1] = apply_deadzone(vals[0])
                            elif prefix == 'z' and len(vals) >= 1:
                                joystick_vel[2] = apply_deadzone(vals[0])

                except Exception as parse_err:
                    pass
    except Exception as e:
        print(f"❌ 串口错误: {e}")

# ================= 目标点更新 =================
def update_target_from_joystick(dt):
    global target_pos
    if target_pos is None: return
        
    delta_pos = joystick_vel * MAX_JOYSTICK_SPEED * dt
    target_pos += delta_pos
    
    target_pos[0] = np.clip(target_pos[0], MIN_POS[0], MAX_POS[0])
    target_pos[1] = np.clip(target_pos[1], MIN_POS[1], MAX_POS[1])
    target_pos[2] = np.clip(target_pos[2], MIN_POS[2], MAX_POS[2])

# ================= 主程序 =================
def main():
    global target_pos, target_quat, running, need_reset, joystick_vel, target_joints, control_mode

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # 初始状态
    data.qpos[:6] = Q_INIT
    if model.nu > 0: data.ctrl[:6] = Q_INIT 
    mujoco.mj_forward(model, data)

    # 初始化 IK 和 FK 两个控制器
    ik_follower = TrajectoryFollower(model, data, site_name="attachment_site",
                                     gain_pos=IK_GAIN_POS, gain_rot=IK_GAIN_ROT,
                                     max_dq=MAX_JOINT_VEL)
    fk_controller = ForwardKinematicsController(model, data)

    target_pos = data.site_xpos[ik_follower.site_id].copy()
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, data.site_xmat[ik_follower.site_id])

    serial_thread = threading.Thread(target=serial_listener_loop, args=("COM4", 115200))
    serial_thread.daemon = True
    serial_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        traj_points = []
        target_traj_points = []
        last_time = time.time()

        while viewer.is_running() and running:
            now = time.time()
            dt = min(0.02, now - last_time)
            last_time = now

            # --- 【复位逻辑】 ---
            if need_reset:
                with viewer.lock():
                    data.qpos[:6] = Q_INIT
                    if model.nu > 0: data.ctrl[:6] = Q_INIT
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    
                    # 复位后同步所有目标点，防止乱飞
                    target_pos = data.site_xpos[ik_follower.site_id].copy()
                    mujoco.mju_mat2Quat(target_quat, data.site_xmat[ik_follower.site_id])
                    target_joints = Q_INIT.copy()
                    
                    traj_points.clear()
                    target_traj_points.clear()
                    joystick_vel.fill(0.0)
                    control_mode = "IK"  # 默认回到 IK 模式
                    
                need_reset = False
                print("🔄 收到 p: 指令，机械臂已瞬间归零！")
                continue 

            # ================= 核心双模式路由 =================
            if control_mode == "IK":
                # 1. 更新摇杆目标
                update_target_from_joystick(dt)

                # 2. 限制超前距离
                current_eef_pos = data.site_xpos[ik_follower.site_id].copy()
                err_vec = target_pos - current_eef_pos
                dist = np.linalg.norm(err_vec)
                
                ik_target_pos = target_pos.copy() 
                MAX_LEAD = 0.03  
                if dist > MAX_LEAD:
                    ik_target_pos = current_eef_pos + (err_vec / dist) * MAX_LEAD

                # 3. 步进 IK 运动学
                ik_follower.step_kinematic(ik_target_pos, target_quat) 
                
                # ⚠️ 关键：将当前的关节角同步回 target_joints，防止切回 FK 时瞬间跳跃
                target_joints = data.qpos[:6].copy()

            elif control_mode == "FK":
                # 1. 直接步进 FK 运动学（从 VOFA+ 滑条取值）
                fk_controller.step_kinematic(target_joints)
                
                # ⚠️ 关键：将当前末端位姿同步回 IK target，防止切回 IK 时瞬间跳跃
                target_pos = data.site_xpos[ik_follower.site_id].copy()
                mujoco.mju_mat2Quat(target_quat, data.site_xmat[ik_follower.site_id])
            # =================================================

            # 记录与渲染轨迹
            traj_points.append(data.site_xpos[ik_follower.site_id].copy())
            if len(traj_points) > 1000: traj_points.pop(0)

            target_traj_points.append(target_pos.copy())
            if len(target_traj_points) > 1000: target_traj_points.pop(0)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                
                # 画红球
                mujoco.mjv_initGeom(viewer.user_scn.geoms[0],
                                    mujoco.mjtGeom.mjGEOM_SPHERE, [0.025,0,0],
                                    target_pos, np.eye(3).flatten(), [1,0,0,0.8])
                viewer.user_scn.ngeom = 1
                
                # 画绿线 (实际轨迹)
                start_real = max(0, len(traj_points) - 400)
                for i in range(start_real, len(traj_points)-1):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break 
                    mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                         mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005,
                                         traj_points[i], traj_points[i+1])
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [0, 1, 0, 1]
                    viewer.user_scn.ngeom += 1

                # 画红线 (目标轨迹) - 仅在 IK 模式下有意义
                if control_mode == "IK":
                    start_target = max(0, len(target_traj_points) - 400)
                    for i in range(start_target, len(target_traj_points)-1):
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                             mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                                             target_traj_points[i], target_traj_points[i+1])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [1, 0, 0, 0.6]
                        viewer.user_scn.ngeom += 1

                viewer.sync()
            time.sleep(0.001)

if __name__ == "__main__":
    main()