import time
import mujoco
import mujoco.viewer

xml_path = "./MujocoEnv/two_wheel_legged_robot.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

render_fps = 100
render_dt = 1.0 / render_fps

with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_start = data.time
    real_start = time.perf_counter()
    last_sim_time = data.time

    while viewer.is_running():
        frame_start = time.perf_counter()

        # 先 sync 一次，让 viewer 里的 reset / perturbation / GUI 操作传给 data
        viewer.sync()

        # 如果 viewer reset 了，data.time 通常会变小，例如从 5.2 回到 0
        if data.time < last_sim_time:
            sim_start = data.time
            real_start = time.perf_counter()
            last_sim_time = data.time
            continue

        # 当前真实时间对应的目标仿真时间
        target_sim_time = sim_start + (time.perf_counter() - real_start)

        # 防止电脑太慢时 while 无限补步
        max_steps_per_frame = 100
        step_count = 0

        while data.time < target_sim_time and step_count < max_steps_per_frame:
            mujoco.mj_step(model, data)
            step_count += 1

        # 如果这一帧已经补了太多步，说明仿真追不上真实时间
        # 重新对齐时间，避免越积越多
        if step_count >= max_steps_per_frame:
            sim_start = data.time
            real_start = time.perf_counter()

        last_sim_time = data.time

        viewer.sync()

        elapsed = time.perf_counter() - frame_start
        if elapsed < render_dt:
            time.sleep(render_dt - elapsed)