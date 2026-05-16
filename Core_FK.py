import numpy as np
import mujoco
from typing import Union, List

class ForwardKinematicsController:
    """
    正运动学（FK）控制器：直接控制机器人的关节角度
    工程级优化：支持任意自由度（DOF）的机械臂，自动处理输入维度校验。
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        # 获取受控执行器的数量（适用于标准串联机械臂）
        self.dof = self.model.nu

    def step_kinematic(self, target_qpos: Union[list, np.ndarray]) -> None:
        """
        纯运动学模式验证：直接修改关节角度，瞬间生效（无视物理碰撞和阻力）
        """
        target = np.asarray(target_qpos, dtype=np.float64)
        
        # 防呆校验：确保输入的关节数量和模型一致
        if len(target) != self.dof:
            raise ValueError(f"[FK 错误] 目标角度维度 ({len(target)}) 与模型执行器数量 ({self.dof}) 不匹配！")

        # 映射到关节位置数组
        self.data.qpos[:self.dof] = target
        
        # 清零动力学状态，防止物理重力产生干扰
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        
        # 仅更新几何学状态，绕过前向动力学计算
        mujoco.mj_kinematics(self.model, self.data)

    def step_physics(self, target_qpos: Union[list, np.ndarray]) -> None:
        """
        物理模式：将目标角度作为位置控制器的指令发送给底层电机
        """
        target = np.asarray(target_qpos, dtype=np.float64)
        
        if len(target) != self.dof:
            raise ValueError(f"[FK 错误] 目标角度维度 ({len(target)}) 与模型执行器数量 ({self.dof}) 不匹配！")
            
        self.data.ctrl[:self.dof] = target
        mujoco.mj_step(self.model, self.data)