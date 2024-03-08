import os  # NOQA
from absl import app, flags, logging  # NOQA

os.environ['OPENAI_API_KEY'] = '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'  # Personal  # NOQA

import matplotlib.pyplot as plt  # NOQA
# plt.switch_backend('agg')  # To avoid the runtime error.

from r2d2.robot_env import RobotEnv  # NOQA
from moka.drivers.driver import Driver  # NOQA
from moka.planners.mouse_click_planner import MouseClickPlanner  # NOQA
from moka.planners.visual_prompt_planner import VisualPromptPlanner  # NOQA
from moka.policies.franka_policy import FrankaMarkPolicy  # NOQA
from moka.utils.config_utils import load_config  # NOQA


FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/moka.yaml', 'Location of config file')
flags.DEFINE_string('data_dir', './data', 'Directory to save the data.')
flags.DEFINE_string('task', 'test', 'Name of the task.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_boolean('debug', False, 'Debugging mode.')

flags.DEFINE_boolean('manual', False, 'Manually clicking the points.')
flags.DEFINE_boolean('skip_confirmation', False, 'Skip confirmation.')
flags.DEFINE_boolean('reuse_segmasks', False, 'Using computed segmasks.')
flags.DEFINE_boolean('incontext', False, 'Using in-context examples.')
flags.DEFINE_boolean('save_incontext', False, 'Saving in-context examples.')


# Table wiping
task_instructions = {
    'table_wiping':
        'Move the eyeglasses onto the yellow cloth and use the brush to sweep the blue snack package to the right side of the table.',  # NOQA

    'ultrasound_cleaning':
        'Use the white ultrasound cleaner to clean the metal watch. The unstrasound cleaner has no lid and can be turned on by pressing the red button.',  # NOQA

    'gift_preparation':
        'Make a pink gift box containing the perfurme bottle. Make sure to put some golden filler beneath the perfume.',  # NOQA

    'laptop_packing':
        'Unplug the charge cable and close the lid of the laptop.',  # NOQA

    'fur_removing':
        'Use the fur remover to swipe the white fur ball down.',  # NOQA

    'grocery_bagging':
        'Put the white soap box and the chips bag into the brown paper bag in the correct order. Be careful that the soap is heavy and can smash the chips if it is on the top.',  # NOQA

    'test':
        'test'

}


def main(_):  # NOQA
    config = load_config(FLAGS.config)

    if FLAGS.manual:
        planner = MouseClickPlanner(
            config=config,
            debug=FLAGS.debug,
        )
    else:
        planner = VisualPromptPlanner(
            config=config,
            debug=FLAGS.debug,
            skip_confirmation=FLAGS.skip_confirmation,
            reuse_segmasks=FLAGS.reuse_segmasks,
            use_incontext_examples=FLAGS.incontext,
            task_name=FLAGS.task,
        )

    policy = FrankaMarkPolicy(
        config=config,
        debug=FLAGS.debug,
    )

    env = RobotEnv(
        action_space="cartesian_position",
        camera_kwargs=dict(
            varied_camera=dict(
                image=True,
                depth=True,
                pointcloud=False,
                concatenate_images=False,
            ),
        ),
    )

    data_collector = Driver(
        env=env,
        policy=policy,
        planner=planner,
        task_instruction=FLAGS.task_instruction,
        save_traj_dir=os.path.join(FLAGS.data_dir, 'saved'),
        save_data=True,
        config=config,
        debug=FLAGS.debug,
        skip_confirmation=FLAGS.skip_confirmation,
    )
    data_collector.collect_trajectory(reset_robot=True)


if __name__ == '__main__':
    app.run(main)
