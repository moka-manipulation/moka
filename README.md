# MOKA: Open-World Robotic Manipulation through Mark-based Visual Prompting
This is the official implementation for the paper "MOKA: Open-World Robotic Manipulation through Mark-based Visual Prompting" (RSS 2024).

[[Paper]](https://arxiv.org/abs/2403.03174) | [[Website]](https://moka-manipulation.github.io) 


## Installation

#### Clone the repository, and navigate to the directory:
```
conda create -n moka python=3.10
conda activate moka
pip install -r requirements.txt
pip install -e .
```
#### Install Other Dependencies

Clone the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) repository and set some environment variables. Follow their instructions to if you want to use Docker.
```
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/
``` 
Then, install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://arxiv.org/abs/2304.02643). Some dependency versions are required to be compatible with MOKA, so please follow the instructions below:
```
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip uninstall roboflow supervision
pip install roboflow supervision==0.15.0
```
Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you run into any issues, please refer to the original repositories for more detailed instructions.

Download DINO and SAM checkpoints, in MOKA root directory:

```
mkdir ckpts && cd ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

#### Set up the robot platform 

Please refer to a customized version of [DROID](https://github.com/FangchenLiu/DROID) and follow the instructions in the repository to install the necessary dependencies.

Before you run MOKA, remember to set the following environment variables:

```
export OPENAI_API_KEY=your_openai_key
```

#### [Optional] Collect data and train robot policies

Convert the stored data to RLDS format by referring to the examples shown in [this repo](https://github.com/FangchenLiu/moka_dataset_builder). This dataset is compatible with the dataloader in [octo](https://github.com/octo-models/octo), 
and can be used to finetune the pre-trained model.

## Usage

Check our [demo](https://github.com/moka-manipulation/moka/blob/main/demo.ipynb) for a quick start on visual prompting.
Try it out for your own tasks!

Run full MOKA pipeline on physical robot using the following command:

```
python franka_eval.py --config /path/to/config.yaml --data_dir /path/to/dataset_to_be_saved
```

For those who are not using the Franka robot and DROID platform, you can modify the `moka/policies/franka_policy.py` for your own setup.


## Acknowledgements
Our project couldn't have been possible without the following great works:
1. The perception pipeline in MOKA is adapted from [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://arxiv.org/abs/2304.02643).
2. The robot platform is built on top of [DROID](https://droid-dataset.github.io/).
3. We get the visual-prompting inspiration from [SoM](https://github.com/microsoft/SoM).
