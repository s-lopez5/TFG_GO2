import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/model_box.xml')
d = mujoco.MjData(m)


mujoco.viewer.launch(m, d)


