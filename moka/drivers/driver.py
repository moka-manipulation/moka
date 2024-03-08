import os
import shutil
import time
from datetime import date

import h5py
import numpy as np
import matplotlib.pyplot as plt  # NOQA
from matplotlib import widgets  # NOQA
from scipy import ndimage

from r2d2.misc.time import time_ms
from r2d2.misc.parameters import (
    moka_version,
    robot_serial_number,
    robot_type,
)
from moka.vision import depth_utils
from r2d2.trajectory_utils.trajectory_writer import TrajectoryWriter

# Prepare Data Folder #
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../moka_data")


def crop_image(image, crop):
    return image[crop[0]:crop[2], crop[1]:crop[3]]


class Driver:
    def __init__(
        self,
        env,
        policy,
        planner,
        config,
        task_instruction=None,
        save_data=True,
        save_traj_dir=None,
        debug=False,
        skip_confirmation=False,
    ):
        self.env = env
        self.planner = planner
        self.task_instruction = task_instruction
        self.policy = policy

        self.config = config

        self.wrist_camera_info = self.get_camera_info(
            self.config.camera.wrist)
        self.primary_camera_info = self.get_camera_info(
            self.config.camera.primary)
        self.secondary_camera_info = self.get_camera_info(
            self.config.camera.secondary)
        self.planner_camera_info = self.get_camera_info(
            self.config.camera.planner)

        self.planner_obs = None
        self.resume_joint_pos = None

        assert config.camera.planner.view == 'left', (
            'The depth image is only available from the left view.')
        self.planner.camera_info = self.planner_camera_info

        self.last_traj_path = None
        self.traj_saved = False
        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        # Make Sure Log Directorys Exist #
        if save_traj_dir is None:
            save_traj_dir = data_dir
        self.success_logdir = os.path.join(
            save_traj_dir, "success", str(date.today())
        )
        self.failure_logdir = os.path.join(
            save_traj_dir, "failure", str(date.today())
        )
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
        self.save_data = save_data

        self.debug = debug
        self.skip_confirmation = skip_confirmation

    def get_camera_info(self, camera_config):
        camera_name = camera_config.serial_number + '_' + camera_config.view
        camera = self.env.camera_reader.get_camera(camera_config.serial_number)
        camera_intrinsics = camera.get_intrinsics()
        camera_extrinsics = self.env.get_side_camera_extrinsics()
        camera_params = {
            'intrinsics': camera_intrinsics[camera_name],
            'extrinsics': camera_extrinsics[camera_name],
        }
        return {
            'name': camera_name,
            'params': camera_params,
            'crop': camera_config.crop,
        }

    def get_planner_observation(self, check=False):

        def show_image():
            if check:
                fig, ax = plt.subplots(ncols=2)
                fig.set_figheight(5)
                fig.set_figwidth(15)
                ax[0].axis('off')
                ax[1].axis('off')
                ax_button1 = plt.axes([0.25, 0.025, 0.1, 0.075])
                ax_button2 = plt.axes([0.65, 0.025, 0.1, 0.075])

            def on_ok_button_clicked(event):
                plt.close()

            def plot():
                # image = np.random.uniform(0, 1, size=[720, 1080, 3])
                # depth = np.random.uniform(0, 1, size=[720, 1080])

                print('Taking images, please wait...')
                self.planner_obs = self._get_planner_observation()

                if check:
                    ax[0].imshow(self.planner_obs['image_data'])
                    ax[1].imshow(self.planner_obs['depth_data'])

                    button_ok = widgets.Button(ax_button1, "OK")
                    button_ok.on_clicked(
                        on_ok_button_clicked)

                    button_retake = widgets.Button(ax_button2, "Retake")
                    button_retake.on_clicked(
                        on_retake_button_clicked)

                    plt.show()

                print('Images updated.')

            # Define button click action
            def on_retake_button_clicked(b):
                plot()

            plot()

        show_image()

    def _get_planner_observation(self):
        assert self.config.camera.depth.average_across_n_frames >= 1

        camera_name = self.planner_camera_info['name']
        camera_params = self.planner_camera_info['params']  # NOQA

        depth_data_list = []
        for i in range(self.config.camera.depth.average_across_n_frames):
            raw_obs = self.env.get_observation()

            depth_data = raw_obs['depth'][camera_name]
            assert depth_data.size > 0, (
                f"Invalid depth shape: {depth_data.shape}")
            depth_data_list.append(depth_data)

        obs = dict()

        obs['camera_intrinsics'] = camera_params['intrinsics']
        obs['camera_extrinsics'] = raw_obs['camera_extrinsics']

        depth_data = depth_utils.preprocess_depth_list(depth_data_list)
        obs['depth_data'] = depth_data

        depth_inpainted = depth_utils.inpaint(depth_data)
        depth_filtered = ndimage.gaussian_filter(
            depth_inpainted,
            sigma=self.config.camera.depth.grad_gaussian_sigma)
        obs['depth_filtered'] = depth_filtered

        # RGB image
        image_data = raw_obs['image'][camera_name][:, :, :3]
        image_data = image_data[:, :, ::-1]  # BGR -> RGB
        obs['image_data'] = image_data

        # Proprioception
        robot_state = raw_obs['robot_state']
        robot_pose = robot_state['cartesian_position']
        gripper_pose = robot_state['gripper_position']
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        # np.save('planner_obs_button.npy', obs)
        return obs

    def get_observation(self):
        obs = self.env.get_observation()
        del obs['depth']

        # Proprioception
        robot_state = obs['robot_state']
        robot_pose = robot_state['cartesian_position']
        gripper_pose = robot_state['gripper_position']
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        obs['crop'] = dict()

        if self.debug:
            print(
                'current ee position, ee pose, gripper position',
                robot_pose[:3],
                robot_pose[3:],
                gripper_pose,
            )

        for camera_info in [self.wrist_camera_info,
                            self.primary_camera_info,
                            self.secondary_camera_info]:
            camera_name = camera_info['name']
            camera_crop = camera_info['crop']
            image_data = obs['image'][camera_name][:, :, :3][:, :, ::-1]
            image_data_cropped = crop_image(image_data, camera_crop)

            obs['image'][camera_name] = image_data_cropped
            obs['crop'][camera_name] = np.array(camera_crop)

        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # fig.set_figheight(10)
        # fig.set_figwidth(15)
        # images = []
        # for camera_info in [self.primary_camera_info,
        #                     self.secondary_camera_info]:
        #     print(image_data.shape, image_data_cropped.shape)
        #     images += [
        #         image_data,
        #         image_data_cropped,
        #     ]
        # num_images = 0
        # for row in ax:
        #     for col in row:
        #         col.imshow(images[num_images])
        #         num_images += 1
        # fig.show()
        #
        # def on_ok_button_clicked(event):
        #     plt.close()
        #
        # def plot():
        #     # image = np.random.uniform(0, 1, size=[720, 1080, 3])
        #     # depth = np.random.uniform(0, 1, size=[720, 1080])
        #
        #     print('Taking images, please wait...')
        #     self.planner_obs = self._get_planner_observation()
        #
        #     ax[0].imshow(self.planner_obs['image_data'])
        #     ax[1].imshow(self.planner_obs['depth_data'])

        return obs

    def prepare_save_directory(self, log_dir, info, subtask_id):
        file_save_dir = os.path.join(
            log_dir, info["time"] + '_' + str(subtask_id)
        )
        if not os.path.isdir(file_save_dir):
            os.makedirs(file_save_dir)
        print('logging to file_save_dir: ', file_save_dir)

        save_filepath = os.path.join(
            file_save_dir, "trajectory.h5"
        )
        recording_folderpath = os.path.join(
            file_save_dir, "recordings"
        )
        context_path = os.path.join(
            file_save_dir, "context.npy"
        )
        if not os.path.isdir(recording_folderpath):
            os.makedirs(recording_folderpath)

        return save_filepath, context_path, recording_folderpath

    def collect_trajectory_with_mouseclick(  # NOQA
        self,
        log_dir=None,
        info=None,
        save_images=False,
        randomize_reset=False,
        reset_robot=True,
    ):
        """
        Collects a robot trajectory.
        - If policy is None, actions will come from the controller
        - If a horizon is given, we will step the environment accordingly
        - Otherwise, we will end the trajectory when the controller tells us to
        """

        # Check Parameters #
        if save_images:
            assert log_dir is not None
            assert info is not None

        recording_folderpath = None
        save_filepath = None
        context_filepath = None

        self.set_trajectory_mode()
        num_subtasks = 0
        subtask_success = False

        # Double check image before collecting data.
        if not self.debug and not self.skip_confirmation:
            obs = self.get_observation()
            fig, ax = plt.subplots(ncols=3)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            ax[0].imshow(obs['image'][self.primary_camera_info['name']])
            ax[1].imshow(obs['image'][self.secondary_camera_info['name']])
            ax[2].imshow(obs['image'][self.wrist_camera_info['name']])
            fig.show()

        while True:
            print('Subtask # %d' % (num_subtasks))

            if num_subtasks == 0:
                # Prepare For Trajectory #
                # if reset_robot:

                self.env.reset(randomize=randomize_reset)
                input('Press [Enter] to continue to next step.')
                self.get_planner_observation()

                self.planner.reset(self.planner_obs, self.task_instruction)

            elif num_subtasks > 0:
                # 1. Save the current joint posisions (+ gripper status)
                # resume_state, _ = self.env.get_state()
                # resume_joint_pos = resume_state['joint_positions']

                # 2. Move back to neutral.
                self.env.reset_to_joint(self.env.reset_joints)

                # 3. Take the image.
                self.get_planner_observation()

                assert self.resume_joint_pos is not None
                # self.planner.reset(self.planner_obs,
                #                    self.task_instruction)

                # 4. Move back to the previous joint positions.
                self.env.reset_to_joint(np.array(self.resume_joint_pos))

                self.planner.current_subtask_index = num_subtasks

            self.planner.current_subtask_index = num_subtasks

            # reset planner
            self.planner.reset_seg_and_keypoints()
            resample_plan = False
            while True:
                planner_obs = self.planner_obs.copy()
                context = self.planner.sample_subtask(
                    planner_obs, t=num_subtasks, request_context=resample_plan)

                if self.skip_confirmation:
                    ch = 'y'
                else:
                    ch = input(
                        'Continue execution without resampling plan? [y/n]')

                if ch in ['y', 'Y', '', '\n']:
                    break
                elif ch in ['n', 'N']:
                    print('resample plan by calling API')
                    resample_plan = True
                else:
                    print('Invalid input. Please enter [y] or [n].')

            obs = self.get_observation()
            self.policy.reset(obs, context, init=(num_subtasks == 0))
            # self.policy.reset(self.planner_obs, context)

            if log_dir:
                assert info is not None
                save_filepath, context_filepath, recording_folderpath = (
                    self.prepare_save_directory(
                        log_dir, info, num_subtasks
                    )
                )
                print('logging to save_filepath: ',
                      save_filepath)
                print('logging to recording_folderpath: ',
                      recording_folderpath)
                traj_writer = TrajectoryWriter(
                    save_filepath, save_images=save_images, exists_ok=False)
                self.env.camera_reader.start_recording(
                    recording_folderpath
                )

            for num_steps in range(self.config.max_subtask_steps):
                if self.debug:
                    input('press enter to continue to next step')

                print('step: ', num_steps)

                control_timestamps = {"step_start": time_ms()}
                obs = self.get_observation()

                # In case we want to stiching the subtasks for future projects.
                obs['subtask_id'] = num_subtasks

                # Get Action
                control_timestamps["policy_start"] = time_ms()
                action, policy_info = self.policy.sample_actions(obs)

                # Regularize Control Frequency #
                control_timestamps["sleep_start"] = time_ms()
                comp_time = time_ms() - control_timestamps["step_start"]
                sleep_left = (1 / self.env.control_hz) - (comp_time / 1000)
                if sleep_left > 0:
                    time.sleep(sleep_left)

                # Step Environment #
                control_timestamps["control_start"] = time_ms()

                action_info = self.env.step(action)
                action_info.update(policy_info)

                # Save Data #
                control_timestamps["step_end"] = time_ms()
                obs["timestamp"]["control"] = control_timestamps
                timestep = {"observation": obs, "action": action_info}
                if save_filepath:
                    traj_writer.write_timestep(timestep)

                print('task_success: ', action_info["task_success"])

                # Check Termination
                end_traj = (num_steps == self.config.max_subtask_steps - 1
                            or action_info["task_success"])

                if end_traj:
                    break

            ch = None
            while True:
                ch = input('Enter the success [y/n]')
                if ch in ['y', 'Y', '', '\n']:
                    subtask_success = True
                    num_subtasks += 1
                    break
                elif ch in ['n', 'N']:
                    subtask_success = False
                    break
                else:
                    print('Invalid input. Please enter [y] or [n].')

            if recording_folderpath:
                print('stop recording last subtask')
                self.env.camera_reader.stop_recording()

            if save_filepath:
                print('close trajectory writer for last subtask')
                traj_writer.close()

            if subtask_success is False:
                # delete saved data
                print('deleting saved data')
                if os.path.exists(save_filepath):
                    os.remove(save_filepath)
                if os.path.exists(recording_folderpath):
                    for filename in os.listdir(recording_folderpath):
                        file_path = os.path.join(
                            recording_folderpath, filename)
                        try:
                            if (os.path.isfile(file_path) or
                                    os.path.islink(file_path)):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s'
                                  % (file_path, e))

                if os.path.exists(context_filepath):
                    # os.remove(context_filepath)
                    os.rename(context_filepath, context_filepath + '_fail.npy')
            else:
                # save context
                print('saving context and planner obs')
                data = {
                    "context": context,
                    "planner_obs": self.planner_obs,
                }
                np.save(context_filepath, data)

            if not subtask_success:
                ch = input('[Optional] To restart the subtask, revert the scene, enter [y] then press [Enter].')  # NOQA
                if ch not in ['y', 'Y', '', '\n']:
                    print('quit data collection')
                    return {"success": False}
            else:
                print('subtask success! saving joint state')
                resume_state, _ = self.env.get_state()
                self.resume_joint_pos = resume_state['joint_positions']

            if num_subtasks >= self.config.max_subtasks:
                self.env.reset()
                # input('Task is done. Press [Enter] after resetting the
                # scene.')  # TODO
                self.resume_joint_pos = None
                break

        return {"success": subtask_success}

    def reset_robot(self, randomize=False):
        self.env._robot.establish_connection()
        self.env.reset(
            randomize=randomize,
        )

    def set_trajectory_mode(self):
        self.env.camera_reader.set_trajectory_mode()

    def prepare_info(self):
        self.last_traj_name = time.asctime().replace(" ", "_")
        info = {}
        info["time"] = self.last_traj_name
        info["robot_serial_number"] = "{0}-{1}".format(
            robot_type, robot_serial_number
        )
        info["version_number"] = moka_version

        if len(self.cam_ids) != 6:
            print(
                "WARNING: User is trying to collect data without all three cameras running!"  # NOQA
            )
        return info

    def collect_trajectory(self, reset_robot=True):
        for num_tasks in range(self.config.max_tasks):
            print('Starting new task # %d' % (num_tasks))
            self._collect_trajectory(reset_robot)

    def _collect_trajectory(self, reset_robot=True):
        info = self.prepare_info()
        # Collect Trajectory ? do we need to establish connection here?
        # remember to reset the robot
        # probably yes as sometimes the connection is unstable
        self.env._robot.establish_connection()

        success_info = self.collect_trajectory_with_mouseclick(
            reset_robot=reset_robot,
            log_dir=self.success_logdir,
            info=info,
        )
        print('success_info:')
        print(success_info)

        # Sort Trajectory #
        # self.traj_succeed = success_info["success"]

        # if self.traj_succeed:
        #     self.last_traj_path = os.path.join(
        #         self.success_logdir, info["time"]
        #     )
        #     os.rename(
        #         os.path.join(self.failure_logdir, info["time"]),
        #         self.last_traj_path,
        #     )

    def change_trajectory_status(self, success=False):
        # will be good for relabeling
        if (self.last_traj_path is None) or (success == self.traj_saved):
            # last trajectiry is not saved or not successful
            # or success condition is the same as last time
            return

        save_filepath = os.path.join(self.last_traj_path, "trajectory.h5")
        traj_file = h5py.File(save_filepath, "r+")
        traj_file.attrs["success"] = success
        traj_file.attrs["failure"] = not success
        traj_file.close()

        if success:
            new_traj_path = os.path.join(
                self.success_logdir, self.last_traj_name
            )
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = True
        else:
            new_traj_path = os.path.join(
                self.failure_logdir, self.last_traj_name
            )
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = False
