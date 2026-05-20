import mujoco

model = mujoco.MjModel.from_xml_path("./MujocoEnv/two_wheel_legged_robot.xml")
data = mujoco.MjData(model)

print(model.nq, model.nv, model.nu, model.nsensordata)