import meshcat
from urdfpy import URDF
import trimesh
m = trimesh.load('/home/yfx/lerobot_piper/lerobot/common/teleoperators/so102_leader/Leader_arm_links.SLDASM/meshes/base_link.STL')
print(m)

# # Create a meshcat visualizer
# vis = meshcat.Visualizer().open()

# # Load your URDF file
# robot = URDF.load('/home/yfx/lerobot_piper/lerobot/common/teleoperators/so102_leader/Leader_arm_links.SLDASM/urdf/Leader_arm_links.SLDASM.urdf')

# # Add the robot to meshcat
# robot_meshcat = robot.show(vis)

# # The browser should pop up automatically, or you can open the link shown in the terminal
