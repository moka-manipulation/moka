"""Base class of planners."""
import copy
import numpy as np
import matplotlib.pyplot as plt  # NOQA
# from scipy import ndimage

from cvp.vision import depth_utils
from cvp.vision import visualization_utils  # NOQA
from cvp.vision.grasp_utils import Grasp2D
from cvp.vision.grasp_utils import AntipodalDepthImageGraspSampler


def visualize_context(obs, context, camera_params):
    fig, ax = plt.subplots(ncols=1)
    ax = [ax]
    fig.set_figheight(5)
    fig.set_figwidth(15)

    ax[0].imshow(obs['image_data'])

    annotation_info = [
        ['ro', 'grasp'],
        ['yo', 'function'],
        ['bo', 'target'],
        ['co', 'pre-contact'],
        ['co', 'post-contact'],
    ]

    points = [
        context['keypoints_2d']['grasp'],
        context['keypoints_2d']['function'],
        context['keypoints_2d']['target'],
        context['waypoints_2d']['pre_contact'][0],
        context['waypoints_2d']['post_contact'][0],
    ]

    for point_id, point in enumerate(points):
        if point is None:
            continue

        ax[0].plot(
            point[0],
            point[1],
            annotation_info[point_id][0])
        ax[0].annotate(
            annotation_info[point_id][1],
            (point[0], point[1]),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center')

        if point_id in [3, 4]:
            ax[0].plot(
                [point[0], points[2][0]],
                [point[1], points[2][1]],
                'c-')

    if context['grasp_vector'] is not None:
        grasp = Grasp2D.from_vector(context['grasp_vector'], camera_params)
        visualization_utils.plot_grasp(ax[0], grasp)

    # Function to be called when the mouse is clicked
    def onclick(event):
        plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)  # nOQA
    plt.show()


class Planner(object):

    def __init__(self,
                 observation_spec=None,
                 action_spec=None,
                 camera_info=None,
                 config=None,
                 debug=False):
        """Initialize."""
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.config = config
        self.debug = debug
        self.camera_info = camera_info
        self.grasp_sampler = AntipodalDepthImageGraspSampler()

        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.image_plot = self.ax.imshow([[0]], cmap='gray')
        # plt.axis('off')
        # plt.title('Visual Marks')

        self.fig = None
        self.ax = None

    def preprocess_image(self, obs):
        """Preprocess the image.

        Args:
            obs: The current observation.

        Return:
            obs: The preprocessed observation.
        """
        crop = self.config.camera.planner.crop
        cropped_image = (
            obs['image_data'][crop[0]:crop[2], crop[1]:crop[3]].copy())
        # flip image vertically
        cropped_image = cropped_image[::-1, ::-1, :]
        return cropped_image

    def reset(self, obs, command):
        """Reset for the new task/episode.

        Args:
            obs: The initial observation.
            command: The command for the final task as a string.
        """
        raise NotImplementedError

    def sample_subtask(self, obs, t):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.

        Return:
            context: A dictionary that contains the context information for the
                current subtask. This will be part of the input to the
                low-level policy/motion planner.
        """
        raise NotImplementedError

    def subtask_done(self, obs, t):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.
            t: The time step index.

        Return:
            True if the current subtask is done (or timeout), False otherwise.
        """
        raise NotImplementedError

    def compute_context_3d(self, obs, context_2d):
        context_3d = copy.deepcopy(context_2d)

        if self.debug:
            print(
                'image observation shape',
                obs['image_data'].shape,
                obs['depth_data'].shape,
                obs['depth_filtered'].shape,
            )

        print('finish sample grasp')

        if context_2d['keypoints_2d']['grasp'] is not None:
            grasp = select_grasp(
                context_2d,
                obs['depth_filtered'],
                self.camera_info['params'],
                self.config.grasp_sampler.max_dist_from_keypoint,
            )
            x, y, z, angle = grasp.as_4dof()

            print(f'grasp_width:{grasp.width}, grasp_angle{angle}, grasp_position: {x, y, z}')  # NOQA
            # print(context_3d['keypoints_3d']['grasp'], [x, y, z])

            context_3d['keypoints_2d']['grasp'] = grasp.center
            context_3d['grasp_yaw'] = angle
            context_3d['grasp_vector'] = grasp.vector

        else:
            context_3d['grasp_yaw'] = 0.
            context_3d['grasp_vector'] = None

        if 'grasp_proposals' in context_2d:
            del context_3d['grasp_proposals']

        # if self.debug:
        #     image = obs['image']
        #     visualization_utils.plot_grasp(grasp)

        context_3d = compute_context_3d(
            context_3d,
            obs['depth_filtered'],
            camera_params=self.camera_info['params'],
            config=self.config
        )

        if self.debug:
            visualize_context(obs, context_3d, self.camera_info['params'])

        return context_3d


def select_grasp(
    context_2d,
    depth,
    camera_params,
    max_dist,
):
    grasp_keypoint = context_2d['keypoints_2d']['grasp']
    keypoint_depth = depth[int(grasp_keypoint[1]), int(grasp_keypoint[0])]

    min_dist = float('inf')
    best_grasp = None
    for grasp_vec in context_2d['grasp_proposals']:
        grasp = Grasp2D.from_vector(grasp_vec, camera_params)
        dist = np.linalg.norm(grasp.center - grasp_keypoint[:2])
        if dist <= min_dist and dist <= max_dist:
            min_dist = dist
            best_grasp = grasp

    if best_grasp is None:
        best_grasp = Grasp2D(center=grasp_keypoint[0:2],
                             angle=0.,
                             depth=keypoint_depth,
                             width=0.07,
                             camera=camera_params)

    return best_grasp


def compute_context_3d(
        context,
        depth,
        camera_params,
        config,
        debug=False):

    def deprojection(point):
        x = point[0]
        y = point[1]
        z = depth[int(y), int(x)]
        hand_xyz = depth_utils.deproject_pixel_to_3d(
            z,
            point,
            camera_params['intrinsics'],
            camera_params['extrinsics'],
            is_worldframe=True)
        hand_xyz[2] = hand_xyz[2] + config.motion.grasp_z_offset
        return hand_xyz, z

    # TODO(kuan): Add the predicted height.

    # keypoints are dict of points
    context['keypoints_3d'] = {}
    context['keypoints_depth'] = {}
    for k in context["keypoints_2d"].keys():
        point = context["keypoints_2d"][k]
        print(k, point)
        if point is None:
            context['keypoints_3d'][k] = None
            context['keypoints_depth'][k] = None
        else:
            (
                context['keypoints_3d'][k],
                context['keypoints_depth'][k],
            ) = deprojection(point)

    # waypoints are dict of lists
    context['waypoints_3d'] = {}
    context['waypoints_depth'] = {}
    for k in context['waypoints_2d'].keys():
        wps = context['waypoints_2d'][k]
        context['waypoints_3d'][k] = []
        context['waypoints_depth'][k] = []
        for wp in wps:
            if wp is not None:
                xyz, d = deprojection(wp)
                context['waypoints_3d'][k].append(xyz)
                context['waypoints_depth'][k].append(d)

    if debug:
        print('waypoints world coordinate', context['waypoints_3d'])
        print('keypoints world coordinate', context['keypoints_3d'])
        print('waypoints world coordinate', context['waypoints_depth'])
        print('keypoints world coordinate', context['keypoints_depth'])

    return context
