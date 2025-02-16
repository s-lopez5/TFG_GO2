import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/global_model.xml')
d = mujoco.MjData(m)

mujoco.viewer.launch(m, d)