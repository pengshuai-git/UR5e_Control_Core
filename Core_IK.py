import numpy as np
import mujoco
from typing import Optional, Union

class TrajectoryFollower:
    """
    机械臂末端轨迹跟踪器（基于阻尼最小二乘法 DLS-IK）
    工程级优化：
    1. 动态雅可比尺寸匹配，支持任意构型机械臂。
    2. 使用 mj_integratePos 替代直接相加，保证四元数位形空间不被破坏。
    3. 预分配高频计算矩阵，降低内存抖动。
    """
    def __init__(self, 
                 model: mujoco.MjModel, 
                 data: mujoco.MjData, 
                 site_name: str, 
                 gain_pos: float = 40.0, 
                 gain_rot: float = 20.0, 
                 dls_lambda: float = 0.01, 
                 max_dq: float = 8.0,
                 deadzone: float = 0.0005):
        
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"模型中未找到 Site: '{site_name}'")
            
        self.gain_pos = gain_pos
        self.gain_rot = gain_rot
        self.dls_lambda = dls_lambda
        self.max_dq = max_dq
        self.deadzone = deadzone
        self.timestep = model.opt.timestep

        # 【性能优化】预分配雅可比矩阵内存，避免高频循环中重复创建
        self._jac = np.zeros((6, self.model.nv))

    def _get_upright_quat(self) -> np.ndarray:
        """生成末端 Z 轴严格垂直向上的四元数"""
        quat = np.zeros(4)
        z_axis = np.array([0.0, 0.0, 1.0])
        x_temp = np.array([1.0, 0.0, 0.0])
        y_axis = np.cross(z_axis, x_temp)
        y_axis /= (np.linalg.norm(y_axis) + 1e-10)
        x_axis = np.cross(y_axis, z_axis)
        mat = np.column_stack((x_axis, y_axis, z_axis))
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    def _compute_ik_twist(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None) -> np.ndarray:
        """计算阻尼最小二乘 IK，返回关节速度指令 (dq)"""
        pos_err = target_pos - self.data.site_xpos[self.site_id]
        
        if target_quat is None:
            target_quat = self._get_upright_quat()
            
        # 使用官方 C-API 进行四元数误差运算
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, self.data.site_xmat[self.site_id])
        inv_quat = np.zeros(4)
        mujoco.mju_negQuat(inv_quat, curr_quat)
        diff_quat = np.zeros(4)
        mujoco.mju_mulQuat(diff_quat, target_quat, inv_quat)
        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, diff_quat, 1.0)

        # 提取当前位形的雅可比矩阵
        mujoco.mj_jacSite(self.model, self.data, self._jac[:3], self._jac[3:], self.site_id)

        # 伪逆求解
        twist = np.concatenate([pos_err * self.gain_pos, rot_err * self.gain_rot])
        A = self._jac @ self._jac.T + self.dls_lambda * np.eye(6)
        dq = self._jac.T @ np.linalg.solve(A, twist)

        # 关节速度限幅
        dq_max = np.max(np.abs(dq))
        if dq_max > self.max_dq:
            dq *= self.max_dq / dq_max
            
        return dq

    def step(self, target_pos: Union[list, np.ndarray], target_quat: Optional[np.ndarray] = None) -> float:
        """
        物理仿真控制步：计算误差并下发速度指令
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        pos_err_dist = float(np.linalg.norm(target_pos - self.data.site_xpos[self.site_id]))
        
        if pos_err_dist > self.deadzone:
            dq = self._compute_ik_twist(target_pos, target_quat)
            # 安全映射：取执行器数量 (nu) 与 速度自由度 (nv) 的最小值
            ctrl_dim = min(self.model.nu, self.model.nv)
            self.data.ctrl[:ctrl_dim] += dq[:ctrl_dim] * self.timestep
            
        mujoco.mj_step(self.model, self.data)
        return float(np.linalg.norm(target_pos - self.data.site_xpos[self.site_id]))

    def step_kinematic(self, target_pos: Union[list, np.ndarray], target_quat: Optional[np.ndarray] = None) -> float:
        """
        纯运动学控制步：绕过动力学引擎，瞬间步进
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        dq = self._compute_ik_twist(target_pos, target_quat)
        
        # 【工程核心】调用官方积分 API，完美处理四元数及复杂关节的更新，严禁直接相加
        mujoco.mj_integratePos(self.model, self.data.qpos, dq, self.timestep)
        
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        
        return float(np.linalg.norm(target_pos - self.data.site_xpos[self.site_id]))

    # =============== 扩展轨迹功能 ===============
    
    def move_to(self, target_pos: np.ndarray, timeout: float = 2.0, pos_tol: float = 0.01, vel_tol: float = 0.5) -> bool:
        start_time = self.data.time
        while self.data.time - start_time < timeout:
            dist = self.step(target_pos)
            vel = np.linalg.norm(self.data.qvel)
            if dist < pos_tol and vel < vel_tol:
                return True
        return False

    def follow_trajectory(self, trajectory: np.ndarray, pos_tol: float = 0.02, vel_tol: float = 0.5, skip_timeout: float = 3.0) -> np.ndarray:
        idx, n_points = 0, len(trajectory)
        traj_record = []
        last_skip_time = self.data.time

        while idx < n_points and self.data.time < 1e6:
            target = trajectory[idx]
            dist = self.step(target)
            vel = np.linalg.norm(self.data.qvel)
            traj_record.append(self.data.site_xpos[self.site_id].copy())

            if dist < pos_tol and vel < vel_tol:
                idx += 1
                last_skip_time = self.data.time
                continue

            if self.data.time - last_skip_time > skip_timeout and idx > 0:
                idx += 1
                last_skip_time = self.data.time
                continue

        return np.array(traj_record)