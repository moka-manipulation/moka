import numpy as np
from r2d2.misc.transformations import euler_to_rmat


def transform_gripper_position(
        grasp_point,
        function_point,
        target_point,
        grasp_euler,
        current_euler,
):

    if function_point is None:
        current_grasp_point = target_point.copy()
        return current_grasp_point
    else:
        R_grasp = euler_to_rmat(grasp_euler)
        R_current = euler_to_rmat(current_euler)

        T_grasp_to_current = np.eye(4)
        T_grasp_to_current[:3, :3] = np.dot(R_current, R_grasp.T)
        T_grasp_to_current[:3, 3] = target_point - np.dot(
            T_grasp_to_current[:3, :3], function_point)

        grasp_point_homogeneous = np.append(grasp_point, 1)
        current_grasp_point_homogeneous = np.dot(
            T_grasp_to_current, grasp_point_homogeneous)
        current_grasp_point = current_grasp_point_homogeneous[:3]

        return current_grasp_point


class FrankaMarkPolicy(object):

    def __init__(self,
                 config,
                 debug=False,
                 ):
        """Initialize."""
        self.config = config
        self.debug = debug

        self.current_phase_idx = None
        self.phases = None

        self.pre_contact_waypoints = None
        self.grasp_point = None
        self.function_point = None
        self.target_point = None
        self.post_contact_waypoints = None

        self.grasp_pose = None
        self.grasp_height = None

        self.prev_gripper_state = None

        self.init_gripper_state = None
        self.init_gripper_pose = None
        self.init_gripper_grip = None

        self.prev_phase = None

        print(self.config)

    def reset(self, observation, context, init=False):  # NOQA
        print('resetting policy')

        self.init_gripper_state = observation['proprio'][:7]
        self.init_gripper_pose = observation['proprio'][:6]
        self.init_gripper_grip = observation['proprio'][6]

        self.grasp_point = context['keypoints_3d']['grasp']
        if self.grasp_point is None:
            self.grasp_pose = None
            self.grasp_height = None
            grasp_yaw = 0.
        else:
            grasp_yaw = (context['grasp_yaw']) % (np.pi) - np.pi / 2

            grasp_euler = np.array([np.pi, 0, grasp_yaw])

            self.grasp_pose = np.concatenate(
                [self.grasp_point, grasp_euler], axis=-1
            )
            self.grasp_height = (
                self.grasp_pose[2] + self.config.motion.grasp_z_offset)

        self.function_point = context['keypoints_3d']['function']

        self.target_point = context['keypoints_3d']['target']

        if (self.grasp_point is not None and self.function_point is None and self.target_point is not None):  # NOQA
            context['target_euler'] = 'forward'
            target_euler = np.array([np.pi, 0, 0])
        else:
            context['target_euler'] = 'forward'
            target_euler = np.array([np.pi, 0, 0])

        if self.target_point is None:
            self.target_pose = None
        else:
            self.target_pose = np.concatenate(
                [self.target_point, target_euler], axis=-1
            )

        self.pre_contact_waypoints = context['waypoints_3d'][
            'pre_contact'
        ]
        pre_contact_euler = target_euler
        self.pre_contact_poses = [
            np.concatenate([pos, pre_contact_euler], axis=-1)
            for pos in self.pre_contact_waypoints]

        self.post_contact_waypoints = context['waypoints_3d'][
            'post_contact'
        ]
        post_contact_euler = target_euler
        self.post_contact_poses = [
            np.concatenate([pos, post_contact_euler], axis=-1)
            for pos in self.post_contact_waypoints]

        # self.pre_contact_height = 'same'
        # self.post_contact_height = 'same'
        # if self.grasp_pose is None:
        #     self.pre_contact_height = 'above'
        #     self.post_contact_height = 'above'

        self.phases = []

        if not init:
            self.phases += [
                'lift_open',
            ]

        if self.grasp_point is None:
            self.phases += [
                'lift_open',
                'grip'
            ]
        else:
            self.phases += [
                'reach_pre_grasp',
                'reach_grasp',
                'grip',
                'lift',
            ]

        if self.function_point is None and self.target_point is not None:
            self.phases += [
                'reach_pre_target',
            ]
            self.phases += [
                'reach_target',
            ]

        else:
            if len(self.pre_contact_waypoints) > 0:
                self.phases += [
                    'reach_pre_motion',
                ]
                self.phases += [
                    f'reach_pre_contact_{i}'
                    for i in range(len(self.pre_contact_waypoints))
                ]

            if self.target_point is not None:
                self.phases += [
                    'reach_target',
                ]

            if len(self.post_contact_waypoints) > 0:
                self.phases += [
                    f'reach_post_contact_{i}'
                    for i in range(len(self.post_contact_waypoints))
                ]

        self.phases += ['release']

        self.prev_phase = None
        self.current_phase_idx = 0
        self.gripper_timer = None
        print('phases: ', self.phases)

    def sample_actions(  # NOQA
        self,
        observation,
    ):
        gripper_state = observation['proprio'][:7]
        gripper_pose = observation['proprio'][:6]
        gripper_grip = observation['proprio'][6]

        current_phase = self.phases[self.current_phase_idx]
        task_success = False

        print('current phase: ', current_phase)
        # print('gripper_grip: ', gripper_grip)

        if current_phase == 'release':
            goal_pose = gripper_pose.copy()
            goal_pose[2] += self.config.motion.release_z_offset
            goal_pose[2] = max(self.config.motion.min_z, goal_pose[2])

            action = self.open_gripper(goal_pose)
            phase_done = gripper_grip < self.config.motion.grip_open_thresh
            if phase_done:
                task_success = True

        elif current_phase == 'grip':
            if self.gripper_timer is None:
                self.gripper_timer = self.config.motion.grip_timeout

            action = self.close_gripper(gripper_pose)
            self.gripper_timer -= 1
            phase_done = (
                self.gripper_timer <= 0 or
                gripper_grip > self.config.motion.grip_close_thresh)
            print('gripper_grip: ', gripper_grip, 'phase_done: ', phase_done)
            if phase_done:
                self.gripper_timer = None

        elif current_phase == 'lift_open':
            goal_pose = self.init_gripper_pose.copy()

            goal_pose[2] = self.config.motion.safe_z
            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 0,
                vel=self.config.motion.high_vel)

        elif current_phase == 'lift':
            if self.grasp_pose is None:
                goal_pose = self.init_gripper_pose.copy()
            else:
                goal_pose = self.grasp_pose.copy()

            goal_pose[2] = self.config.motion.safe_z
            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1,
                vel=self.config.motion.high_vel)

        elif current_phase == 'reach_pre_grasp':
            grasp_pose = self.grasp_pose.copy()
            grasp_pose[2] = self.config.motion.safe_z
            action, phase_done = self.move_to(
                grasp_pose, gripper_pose, gripper_grip,
                vel=self.config.motion.high_vel)
            print('reach pre-grasp', grasp_pose)

        elif current_phase == 'reach_grasp':
            grasp_pose = self.grasp_pose.copy()
            grasp_pose[2] = self.grasp_height
            action, phase_done = self.move_to(
                self.grasp_pose, gripper_pose, gripper_grip,
                vel=self.config.motion.low_vel)
            print('reach grasp', self.grasp_pose)
            print('action: ', action)

        elif current_phase == 'reach_pre_target':
            pos = self.target_pose[0:3].copy()
            pos[2] = self.config.motion.safe_z
            if self.grasp_height is not None:
                pos[2] = max(pos[2], self.grasp_height)

            euler = self.target_pose[3:6].copy()

            if self.grasp_pose is not None:
                pos = transform_gripper_position(
                    grasp_point=self.grasp_pose[0:3],
                    function_point=self.function_point,
                    target_point=pos,
                    grasp_euler=self.grasp_pose[3:6],
                    current_euler=euler)

            print('reach pre-target', pos)
            goal_pose = np.concatenate([pos, euler], axis=-1)
            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1,
                vel=self.config.motion.high_vel)
            print('action: ', action)

        elif current_phase == 'reach_pre_motion':
            waypoint_idx = 0
            pos = self.pre_contact_waypoints[waypoint_idx].copy()
            pos[2] = self.config.motion.safe_z

            euler = self.pre_contact_poses[waypoint_idx][3:6].copy()

            if self.grasp_pose is not None:
                pos = transform_gripper_position(
                    grasp_point=self.grasp_pose[0:3],
                    function_point=self.function_point,
                    target_point=pos,
                    grasp_euler=self.grasp_pose[3:6],
                    current_euler=euler)

            print('reach pre-motion', pos)
            goal_pose = np.concatenate([pos, euler], axis=-1)
            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1,
                vel=self.config.motion.high_vel)
            print('action: ', action)

        elif 'pre_contact' in current_phase:
            waypoint_idx = int(current_phase.split('_')[-1])
            pos = self.pre_contact_waypoints[waypoint_idx].copy()

            if self.pre_contact_height == 'same':
                if self.grasp_height is not None:
                    pos[2] = max(pos[2], self.grasp_height)
                vel = self.config.motion.low_vel
            elif self.pre_contact_height == 'above':
                pos[2] = self.config.motion.safe_z
                vel = self.config.motion.high_vel
            else:
                raise ValueError('Unrecognized height: %s'
                                 % (self.pre_contact_height))

            euler = self.pre_contact_poses[waypoint_idx][3:6].copy()

            if self.grasp_pose is not None:
                pos = transform_gripper_position(
                    grasp_point=self.grasp_pose[0:3],
                    function_point=self.function_point,
                    target_point=pos,
                    grasp_euler=self.grasp_pose[3:6],
                    current_euler=euler)

            print('pre-contact waypoint', pos)
            goal_pose = np.concatenate([pos, euler], axis=-1)

            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1,
                vel=vel)
            print('action: ', action)

        elif current_phase == 'reach_target':
            pos = self.target_pose[0:3].copy()
            pos[2] += self.config.motion.grasp_z_offset
            if self.grasp_height is not None:
                pos[2] = max(pos[2], self.grasp_height)

            euler = self.target_pose[3:6].copy()

            if self.grasp_pose is not None:
                pos = transform_gripper_position(
                    grasp_point=self.grasp_pose[0:3],
                    function_point=self.function_point,
                    target_point=pos,
                    grasp_euler=self.grasp_pose[3:6],
                    current_euler=euler)

            print('reach target', pos)

            goal_pose = np.concatenate([pos, euler], axis=-1)

            if self.pre_contact_height == 'same':
                # Moving horizontally.
                vel = self.config.motion.high_vel
            elif self.pre_contact_height == 'above':
                # Moving downward.
                vel = self.config.motion.low_vel
            else:
                raise ValueError('Unrecognized pre_contact_height: %s'
                                 % (self.pre_contact_height))

            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1,
                vel=vel,
                pos_thresh=self.config.motion.target_pos_thresh)

        elif 'post_contact' in current_phase:
            waypoint_idx = int(current_phase.split('_')[-1])

            pos = self.post_contact_waypoints[waypoint_idx]

            if self.post_contact_height == 'same':
                if self.grasp_height is not None:
                    pos[2] = max(pos[2], self.grasp_height)
                vel = self.config.motion.high_vel
            elif self.post_contact_height == 'above':
                pos[2] = self.config.motion.safe_z
                vel = self.config.motion.high_vel
            else:
                raise ValueError('Unrecognized height: %s'
                                 % (self.post_contact_height))

            euler = self.post_contact_poses[waypoint_idx][3:6]

            if self.grasp_pose is not None:
                pos = transform_gripper_position(
                    grasp_point=self.grasp_pose[0:3],
                    function_point=self.function_point,
                    target_point=pos,
                    grasp_euler=self.grasp_pose[3:6],
                    current_euler=euler)

            goal_pose = np.concatenate([pos, euler], axis=-1)
            action, phase_done = self.move_to(
                goal_pose, gripper_pose, 1, vel=vel)

        else:
            raise ValueError('Unrecognized phase: %r' % (current_phase))

        if (self.prev_phase != current_phase or
                self.prev_gripper_state is None or
                (np.linalg.norm(
                    self.prev_gripper_state[:6] - gripper_state[:6])
                 >= self.config.motion.target_pos_thresh
                 and current_phase != 'grip')):  # NOQA
            self.stuck_countdown = 5
        else:
            if current_phase != 'grip':
                print('Warning: The robot gripper is stuck!')
                self.stuck_countdown -= 1

        print('stuck_countdown: ', self.stuck_countdown,
              self.prev_gripper_state, gripper_state)

        if self.stuck_countdown <= 0:
            phase_done = True

        self.prev_gripper_state = gripper_state
        self.prev_phase = current_phase

        if phase_done:
            if self.current_phase_idx < len(self.phases) - 1:
                self.current_phase_idx += 1

        action = np.array(action, dtype=np.float32)
        return action, {'task_success': task_success}

    # franka uses absolute position control
    def move_to(self, target_pose, gripper_pose, grip, vel,
                pos_thresh=None, euler_thresh=None):

        if pos_thresh is None:
            pos_thresh = 0.5 * vel

        if euler_thresh is None:
            euler_thresh = 0.5 * self.config.motion.max_angvel

        target_pose = target_pose.copy()
        target_pose[0] += self.config.motion.gripper_xy_offset[0]
        target_pose[1] += self.config.motion.gripper_xy_offset[1]
        target_pose[2] = max(self.config.motion.min_z, target_pose[2])
        d_pos = target_pose[:3] - gripper_pose[:3]

        # get delta translation
        if np.linalg.norm(d_pos) < pos_thresh:
            pos_done = True
            target_pos = gripper_pose[0:3]
        else:
            pos_done = False
            translation = d_pos / np.linalg.norm(d_pos)
            translation *= vel
            target_pos = gripper_pose[:3] + translation
            target_pos[2] = max(self.config.motion.min_z, target_pos[2])

        # get delta rotation
        d_euler = target_pose[3:6] - gripper_pose[3:6]
        d_euler[0] = (d_euler[0] - np.pi) % (2 * np.pi) - np.pi
        d_euler[1] = (d_euler[1] - np.pi / 2) % (np.pi) - np.pi / 2
        d_euler[2] = (d_euler[2] - np.pi) % (2 * np.pi) - np.pi

        if np.max(np.abs(d_euler)) < euler_thresh:
            euler_done = True
            target_euler = gripper_pose[3:6]
        else:
            euler_done = False
            target_euler = np.array(target_pose[3:6], dtype=np.float32)

        action = np.concatenate(
            [target_pos, target_euler, [grip]], axis=-1)

        done = (pos_done and euler_done)

        return action, done

    def open_gripper(self, gripper_pose):
        action = np.concatenate([gripper_pose, np.array([0])])
        return action

    def close_gripper(self, gripper_pose):
        action = np.concatenate([gripper_pose, np.array([1])])
        return action
