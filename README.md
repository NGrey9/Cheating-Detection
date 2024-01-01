# Using MMAction2 for Cheating Detection in Examination

## 📖 Introduction

MMAction2 is an open-source toolbox for video understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.com/) project.


<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Skeleton-based Spatio-Temporal Action Detection and Action Recognition Results on Kinetics-400</p>
</div>

## 🛠️ Installation

MMAction2 depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), [MMDetection](https://github.com/open-mmlab/mmdetection) (optional) and [MMPose](https://github.com/open-mmlab/mmpose) (optional).

Please refer to [install.md](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) for detailed instructions.
<details close>
<summary>Quick instructions</summary>

```shell
git clone https://github.com/open-mmlab/mmaction2.git
conda create --name openmmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

</details>


## 🛠️ Modifying


To implement my model to Cheating Detection using camera,
you need to modify something in your mmaction2 directory:

- Change original mmaction2/demo/demo_skeleton.py to be the same as my mmaction2/demo/demo_skeleton.py
- Change original mmaction2/mmaction/apis/inference.py to be the same as my mmaction2/mmaction/apis/inference.py
- Change original mmaction2/utils/misc.py to be the same as my mmaction2/utils/misc.py

Then you should modify the labels.txt 

```shell
lean forward
tease the person above
bent down to the ground to pick up documents
see documents below the exam
lean to the sides
no cheating
no cheating
```
You need to use mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py to be default config for Recognition,
and you should set num_classes=7 

Finally, you run:
```shell
python mmaction2/demo/demo_skeleton.py [your-video-input-path] [your-output-path] --checkpoint [checkpoint-path] --label-map [labels-text-file] --config [config-file-path] --device ['cpu' or 'cuda:0']
```
to generate the output.

## 🎁 Results
<div align="center">
  <img src="https://raw.githubusercontent.com/NGrey9/Cheating-Detection/main/assets/test.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">My Recognition Model Results</p>
</div>




