import time
import numpy as np

import mujoco
import mujoco.viewer

# model_path = "robot_description/torque_ctrl.xml"
model_path = "robot_description/position_ctrl.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl = data.qpos[7:]
    ctrl_0 = data.ctrl.copy()

    wall = time.monotonic()
    while viewer.is_running():
        t = time.monotonic() - wall
        st = np.sin(t) * 0.1

        # set control
        data.ctrl[0] = ctrl_0[0] - st
        data.ctrl[1] = ctrl_0[1] + st
        data.ctrl[2] = ctrl_0[2] - 2*st

        data.ctrl[3] = ctrl_0[3] + st
        data.ctrl[4] = ctrl_0[4] - st
        data.ctrl[5] = ctrl_0[5] + 2*st

        data.ctrl[6] = ctrl_0[6] - st
        data.ctrl[7] = ctrl_0[7] + st
        data.ctrl[8] = ctrl_0[8] - 2*st

        data.ctrl[9] = ctrl_0[9] + st
        data.ctrl[10] = ctrl_0[10] - st
        data.ctrl[11] = ctrl_0[11] + 2*st

        # step simulation
        while data.time <= t:
            mujoco.mj_step(model, data)

        viewer.sync()
