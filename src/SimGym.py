import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class Go2Env(gym.Env):
    def __init__(self, render_mode=None):
        super(Go2Env, self).__init__()
        
        # Cargar el modelo MuJoCo
        self.model = mujoco.MjModel.from_xml_path("/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml")
        self.data = mujoco.MjData(self.model)

        # Acción: torques o posiciones deseadas en cada articulación
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Observación: posiciones + velocidades articulares, etc.
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Aplicar acción (ej: torques)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        # Define tu función de recompensa aquí
        reward = self._compute_reward()
        terminated = False  # Cambia esto si hay condiciones de fallo
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward(self):
        # Ejemplo: recompensa por avanzar hacia delante
        forward_velocity = self.data.qvel[0]  # Ajusta al DOF correcto
        return forward_velocity

    def render(self):
        if self.render_mode == "human":
            viewer = mujoco.viewer.launch_passive(self.model, self.data)