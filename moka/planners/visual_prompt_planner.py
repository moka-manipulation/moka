# from openai import OpenAI
# client = OpenAI()
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np  # noqa
import json
from PIL import Image

from moka.planners.visual_prompt_utils import (
    annotate_visual_prompts,
    annotate_motion,
    propose_candidate_keypoints,
    request_motion,
    request_plan,
)
from moka.planners.planner import Planner
from moka.vision import visualization_utils  # NOQA
from moka.vision.grasp_utils import Grasp2D
from moka.vision.segmentation import get_segmentation_masks
from moka.vision.segmentation import get_scene_object_bboxes


pp = pprint.PrettyPrinter(indent=4)


OFFSET = 10

# --------------------------------
# Planner
# --------------------------------


class VisualPromptPlanner(Planner):
    def __init__(
        self,
        config,
        prompts=None,
        debug=False,
        skip_confirmation=False,
        use_center=False,
        reuse_segmasks=False,
        use_incontext_examples=False,
        num_incontext_examples=3,
        task_name=None,
    ):
        """Initialize."""
        super().__init__(config=config, debug=debug)
        self.config = config
        self.debug = debug
        self.skip_confirmation = skip_confirmation
        self.use_center = use_center
        self.reuse_segmasks = reuse_segmasks
        self.use_incontext_examples = use_incontext_examples
        self.num_incontext_examples = num_incontext_examples

        if prompts is None:
            self.prompts = self._load_prompts()
        else:
            self.prompts = prompts

        self.plan = None
        self.current_subtask_index = None
        self.current_subtask_timeout = None
        self.all_object_names = []  # Currently unused.
        self.segmasks = None
        self.saved_segmasks = None
        self.context_2d = dict()
        self.task_instruction = None
        self.task_name = task_name

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

    def transform_points(self, context_2d):
        def transform(point):
            # flip vertically
            crop = self.config.camera.planner.crop
            crop_shape = (crop[2] - crop[0], crop[3] - crop[1])
            # image_shape = obs['image_data'].shape
            # point = np.array(point)
            point[1] = crop_shape[0] - point[1]
            point[0] = crop_shape[1] - point[0]
            # add crop
            point[0] += self.config.camera.planner.crop[1]
            point[1] += self.config.camera.planner.crop[0]
            return point

        # transform keypoints to the original image
        for key in context_2d['keypoints_2d'].keys():
            if context_2d['keypoints_2d'][key] is not None:
                kp = context_2d['keypoints_2d'][key].copy()
                context_2d['keypoints_2d'][key] = transform(kp)

        # transform waypoints to the original image
        for key in context_2d['waypoints_2d'].keys():
            wps = context_2d['waypoints_2d'][key]
            for i in range(len(wps)):
                if wps[i] is not None:
                    wps[i] = transform(wps[i].copy())

        return context_2d

    def reset(self, obs, task_instruction):
        """Reset for the new task/episode.

        Args:
            obs: The initial observation.
            task_instruction: The command for the final task as a string.
        """
        processed_img = self.preprocess_image(obs)
        obs_image = Image.fromarray(processed_img).convert('RGB')
        self.task_instruction = task_instruction
        # visualize
        # print('plotting planner image')
        # plt.imshow(obs_image)
        # plt.axis('off')
        # plt.show()

        self.plan = None
        # if file exists, load the plan from json file
        # get current directory

        if self.task_name is None:
            task_name = task_instruction.replace(' ', '_')
        else:
            task_name = self.task_name
        plan_path = self.config.log_dir + f'/{task_name}/plan.json'
        os.makedirs(self.config.log_dir + f'/{task_name}', exist_ok=True)
        print('Saving to ', self.config.log_dir + f'/{task_name}')

        if os.path.exists(plan_path):
            with open(plan_path, 'r') as f:
                self.plan = json.load(f)
                print('loaded plan from json file')

        else:
            # otherwise, request plan from API
            print('requesting plan from API')
            while self.plan is None:
                self.plan = request_plan(
                    task_instruction,
                    obs_image,
                    plan_with_obs_image=self.config.plan_with_obs_image,
                    prompts=self.prompts,
                    debug=self.debug,
                )
                print('Plan until get valid answer...')
                print('Plan: ')
                pp.pprint(self.plan)

            # save plan to json file
            with open(plan_path, 'w') as f:
                json.dump(self.plan, f)

        self.current_subtask_index = 0
        self.total_subtasks = len(self.plan)
        self.all_object_names = []

        if self.saved_segmasks is None:
            self.saved_segmasks = [None] * self.total_subtasks

        for subtask in self.plan:
            if (subtask['object_grasped'] != '' and
                    subtask['object_grasped'] not in self.all_object_names):
                self.all_object_names.append(subtask['object_grasped'])

            if (subtask['object_unattached'] != '' and
                    subtask['object_unattached'] not in self.all_object_names):
                self.all_object_names.append(subtask['object_unattached'])

        print('all_object_names:', self.all_object_names)

    def get_segmentation_and_keypoints(self, obs_image):
        if (self.reuse_segmasks and
                self.saved_segmasks[self.current_subtask_index] is not None):
            self.segmasks = self.saved_segmasks[self.current_subtask_index]
        else:
            boxes, logits, phrases = get_scene_object_bboxes(
                obs_image, self.all_object_names,
                visualize=True,
                logdir=self.config.log_dir)

            self.segmasks = get_segmentation_masks(
                obs_image, self.all_object_names, boxes, logits, phrases,
                visualize=True,
                logdir=self.config.log_dir)

            self.saved_segmasks[self.current_subtask_index] = self.segmasks

        subtask = self.plan[self.current_subtask_index]

        # get keypoints from segmentation
        self.candidate_keypoints = propose_candidate_keypoints(
            subtask,
            self.segmasks,
            num_samples=self.config.num_candidate_keypoints)

    def reset_seg_and_keypoints(self):
        self.segmasks = None
        self.candidate_keypoints = None

    def sample_subtask(self, obs, t=0, request_context=False):  # NOQA
        """Reset for the new task/episode.

        Args:
            obs: The current observation.

        Return:
            context: A dictionary that contains the context information for the
                current subtask. This will be part of the input to the
                low-level policy/motion planner.
        """
        assert self.current_subtask_index >= 0

        processed_img = self.preprocess_image(obs)
        obs_image = Image.fromarray(processed_img).convert('RGB')

        subtask = self.plan[self.current_subtask_index]
        print('current subtask:', subtask)

        if self.segmasks is None or self.candidate_keypoints is None:
            print('recompute segmentation and keypoints')
            self.get_segmentation_and_keypoints(obs_image)

        candidate_keypoints = self.candidate_keypoints.copy()

        annotated_image = annotate_visual_prompts(
            obs_image,
            candidate_keypoints,
            waypoint_grid_size=self.config.waypoint_grid_size,
            log_dir=self.config.log_dir,
        )

        assert self.task_instruction is not None

        if self.task_name is None:
            task_name = self.task_instruction.replace(' ', '_') + '_'
        else:
            task_name = self.task_name

        subtask_id = 'subtask_' + str(self.current_subtask_index)
        os.makedirs(
            self.config.log_dir + f'/{task_name}/{subtask_id}',
            exist_ok=True)
        print('Saving to ', self.config.log_dir + f'/{task_name}/{subtask_id}')

        context_path = self.config.log_dir + f'/{task_name}/{subtask_id}/context.json'  # NOQA
        obs_image_path = self.config.log_dir + f'/{task_name}/{subtask_id}/obs_image.jpg'  # NOQA
        annotated_image_path = self.config.log_dir + f'/{task_name}/{subtask_id}/annotated_image.jpg'  # NOQA
        motion_image_path = self.config.log_dir + f'/{task_name}/{subtask_id}/motion_image.jpg'  # NOQA

        if (
                os.path.exists(context_path) and
                not request_context):

            with open(context_path, 'r') as f:
                context_2d_json = json.load(f)

            context_2d, _, motion_image = request_motion(
                subtask,
                obs_image,
                annotated_image,
                candidate_keypoints,
                waypoint_grid_size=self.config.waypoint_grid_size,
                prompts=self.prompts,
                debug=True,
                loaded_context=context_2d_json,
                use_center=self.use_center,
                log_dir=self.config.log_dir,
                add_caption=False,
            )
            obs_image.save(obs_image_path)
            annotated_image.save(annotated_image_path)
            motion_image.save(motion_image_path)

            print('loaded context from json file')
            print(f'context 2d for subtask {self.current_subtask_index}:',
                  context_2d)

        # otherwise, request context from API
        else:
            if self.task_name is None:
                task_name = self.task_instruction.replace(' ', '_') + '_'
            else:
                task_name = self.task_name

            subtask_id = 'subtask_' + str(self.current_subtask_index)

            if self.use_incontext_examples:
                example_images = []
                example_responses = []
                for example_idx in range(self.num_incontext_examples):
                    example_context_path = f'incontext/{task_name}/{subtask_id}/{example_idx}/context.json'  # NOQA
                    example_image_path = f'incontext/{task_name}/{subtask_id}/{example_idx}/annotated_image.jpg'  # NOQA
                    annotated_image = Image.open(example_image_path)
                    example_images.append(annotated_image)

                    with open(example_context_path, 'r') as f:
                        context_2d_json = json.load(f)
                    example_response = json.dumps(context_2d_json, indent=4)
                    example_response = f"""
```json
{example_response}
```
"""
                    example_responses.append(example_response)
                print('Using in-context examples: ')
                print(example_responses)

            else:
                example_images = None
                example_responses = None

            print('requesting context from API')
            context_2d, context_2d_json = None, None
            while context_2d is None:
                context_2d, context_2d_json, motion_image = request_motion(
                    subtask,
                    obs_image,
                    annotated_image,
                    candidate_keypoints,
                    waypoint_grid_size=self.config.waypoint_grid_size,
                    prompts=self.prompts,
                    use_center=self.use_center,
                    example_images=example_images,
                    example_responses=example_responses,
                    debug=True,
                    log_dir=self.config.log_dir,
                    add_caption=False,
                )
                print('context 2d until get valid answer:', context_2d)
                obs_image.save(obs_image_path)
                annotated_image.save(annotated_image_path)
                motion_image.save(motion_image_path)

            if not self.use_incontext_examples:
                # save plan to json file
                if request_context:
                    ch = 'y'
                else:
                    ch = input('press enter to save context')

                if ch == '' or ch == 'y' or ch == 'Y' or ch == '\n':
                    with open(context_path, 'w') as f:  # NOQA
                        json.dump(context_2d_json, f)
                    obs_image.save(obs_image_path)
                    annotated_image.save(annotated_image_path)
                    print('context saved')
                else:
                    print('context not saved')

        context_2d = self.transform_points(context_2d)

        if context_2d['keypoints_2d']['grasp'] is None:
            grasp_proposals = None
            crop = None
        else:

            grasp_object = subtask['object_grasped']
            bbox = self.segmasks[grasp_object]['bbox'].copy()

            crop = self.config.camera.planner.crop
            crop_shape = (crop[2] - crop[0], crop[3] - crop[1])

            bbox[0] = max(0, crop_shape[0] - bbox[0] + self.config.camera.planner.crop[0] + OFFSET)  # NOQA
            bbox[1] = max(0, crop_shape[1] - bbox[1] + self.config.camera.planner.crop[1] + OFFSET)  # NOQA
            bbox[2] = max(0, crop_shape[0] - bbox[2] + self.config.camera.planner.crop[0] - OFFSET)  # NOQA
            bbox[3] = max(0, crop_shape[1] - bbox[3] + self.config.camera.planner.crop[1] - OFFSET)  # NOQA

            new_crop = [bbox[2], bbox[3], bbox[0], bbox[1]]
            print('new crop:', new_crop)

            # self.fig, self.ax = plt.subplots(ncols=2)
            # self.fig.set_figheight(5)
            # self.fig.set_figwidth(15)
            #
            # # Display the image
            # self.ax[0].imshow(obs['image_data'])
            # self.ax[1].imshow(obs['depth_data'])
            #
            # y1, x1, y2, x2 = new_crop
            #
            # self.ax[0].plot([x1, x1], [y1, y2], color='red')
            # self.ax[0].plot([x2, x2], [y1, y2], color='red')
            # self.ax[0].plot([x1, x2], [y2, y2], color='red')
            # self.ax[0].plot([x1, x2], [y1, y1], color='red')
            #
            # self.ax[1].plot([x1, x1], [y1, y2], color='red')
            # self.ax[1].plot([x2, x2], [y1, y2], color='red')
            # self.ax[1].plot([x1, x2], [y2, y2], color='red')
            # self.ax[1].plot([x1, x2], [y1, y1], color='red')

            # show the cropped image

            try:
                grasp_proposals = self.grasp_sampler.sample_grasp(
                    obs['depth_data'],
                    obs['depth_filtered'],
                    self.camera_info['params'],
                    crop=new_crop,
                    num_samples=self.config.grasp_sampler.num_samples,
                )
            except Exception:
                print('Warning: No grasp proposal was detected!')
                grasp_proposals = []

        context_2d['grasp_proposals'] = grasp_proposals

        context = self.compute_context_3d(obs, context_2d)

        if not self.skip_confirmation:
            self._visualization(obs, context_2d, crop)

        for k, v in context.items():
            print(f"{k}: {v}")  # display context

        return context

    def subtask_done(self, obs, t):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.
            t: The time step index.

        Return:
            True if the full task is done, False otherwise.
        """
        return True


    def _visualization(self, obs, context, crop):  # NOQA
        self.fig, self.ax = plt.subplots(ncols=2)
        self.fig.set_figheight(5)
        self.fig.set_figwidth(15)

        self.points = [context['keypoints_2d']['grasp'],
                       context['keypoints_2d']['function'],
                       context['keypoints_2d']['target'],
                       context['waypoints_2d']['pre_contact'][0],
                       context['waypoints_2d']['post_contact'][0]]

        grasp_proposals = context['grasp_proposals']

        # Display the image
        self.ax[0].imshow(obs['image_data'])
        self.ax[1].imshow(obs['depth_data'])

        if crop is not None:
            y1, x1, y2, x2 = crop
        else:
            y1, x1, y2, x2 = self.config.grasp_sampler.crop

        self.ax[0].plot([x1, x1], [y1, y2], color='red')
        self.ax[0].plot([x2, x2], [y1, y2], color='red')
        self.ax[0].plot([x1, x2], [y2, y2], color='red')
        self.ax[0].plot([x1, x2], [y1, y1], color='red')

        self.ax[1].plot([x1, x1], [y1, y2], color='red')
        self.ax[1].plot([x2, x2], [y1, y2], color='red')
        self.ax[1].plot([x1, x2], [y2, y2], color='red')
        self.ax[1].plot([x1, x2], [y1, y1], color='red')

        if grasp_proposals is not None:
            for grasp_vec in grasp_proposals:
                grasp = Grasp2D.from_vector(
                    grasp_vec, self.camera_info['params'])
                visualization_utils.plot_grasp(self.ax[0], grasp)
                visualization_utils.plot_grasp(self.ax[1], grasp)

        annotation_info = [
            ['ro', 'grasp'],
            ['yo', 'function'],
            ['bo', 'target'],
            ['co', 'pre-contact'],
            ['co', 'post-contact'],
        ]

        for point_id, point in enumerate(self.points):
            if point is not None:
                self.ax[0].plot(
                    point[0],
                    point[1],
                    annotation_info[point_id][0])
                self.ax[0].annotate(
                    annotation_info[point_id][1],
                    (point[0], point[1]),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha='center')

                if point_id in [3, 4]:
                    self.ax[0].plot(
                        [point[0], self.points[2][0]],
                        [point[1], self.points[2][1]],
                        'c-')

        plt.show()

    def _load_prompts(self):
        """Load prompts from the file.
        """
        prompts = dict()
        prompt_dir = os.path.join(
            self.config.prompt_root_dir, self.config.prompt_name)
        print('prompt_dir: ', prompt_dir)
        for filename in os.listdir(prompt_dir):
            path = os.path.join(prompt_dir, filename)
            if os.path.isfile(path) and path[-4:] == '.txt':
                with open(path, 'r') as f:
                    value = f.read()
                key = filename[:-4]
                prompts[key] = value
        return prompts


if __name__ == "__main__":
    import yaml
    # import matplotlib.pyplot as plt
    from easydict import EasyDict as edict

    config_filename = 'configs/visual_prompt_planner.yaml'
    with open(config_filename, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.SafeLoader)
        config = edict(config)

    planner = VisualPromptPlanner(None, None, config, debug=True)

    obs_image = Image.open('data/table_cleaning.jpg').convert('RGB')
    obs = {'image': obs_image}
    plt.imshow(obs_image)
    plt.axis('off')
    print('image size:', (obs_image.size))
    plt.show()

    task_instruction = 'Sweep the jelly packs to the right side of the table and pull the bowl backward using the hammer.'  # NOQA
    print('Task: ', task_instruction)

    planner.reset(obs, task_instruction)
    context = planner.sample_subtask(obs, t=0)
    annotate_motion(obs_image, context)
    plt.imshow()
