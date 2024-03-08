from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import (
    DataCollecter,
)
from r2d2.user_interface.gui import RobotGUI


def camera_calibration():
    # Make the robot env
    env = RobotEnv(calibration_mode=True)
    controller = VRPolicy(spatial_coeff=2)

    # Make the data collector
    data_collector = DataCollecter(
        env=env, controller=controller
    )

    # Make the GUI
    user_interface = RobotGUI(robot=data_collector)

def data_collection():
    # Make the robot env
    env = RobotEnv(calibration_mode=False)
    controller = VRPolicy(spatial_coeff=2)

    # Make the data collector
    data_collector = DataCollecter(
        env=env, controller=controller
    )

    # Make the GUI
    user_interface = RobotGUI(robot=data_collector)


if __name__ == '__main__':
    data_collection()
    # camera_calibration()
