![](./assets/logo.png)
<h1> What Do You See in Vehicle? Comprehensive Vision Solution for In-Vehicle Gaze Estimation </h1> 

<i><a href="https://yihua.zone/">Yihua Cheng </a>, Yaning Zhu, Zongji Wang, Hongquan Hao, Yongwei Liu, Shiqing Cheng, Xi Wang, <a href="https://hyungjinchang.wordpress.com"> Hyung Jin Chang</a>,  CVPR 2024</i>

<a href='https://arxiv.org/abs/2403.15664'> <img src='https://img.shields.io/badge/ArXiv-PDF-red' style="vertical-align:middle;"> </a>  <a href='https://yihua.zone/work/ivgaze/'> <img src='https://img.shields.io/badge/Demo-Project%20Page-red' style="vertical-align:middle;"> </a>
<a href='https://www.birmingham.ac.uk/'> <img src='https://img.shields.io/badge/UK-Unversity%20of%20Birmingham-red' style="vertical-align:middle;"> </a> 
 

## Description
This repository provides offical code of the paper titled *What Do You See in Vehicle? Comprehensive Vision Solution for In-Vehicle Gaze Estimation*, accepted at CVPR24.
Our contribution includes:
- We provide a dataset **IVGaze** collected on vehicles containing 44k images of 125 subjects.
- We propose a gaze pyramid transformer (GazePTR) that leverages transformer-based multilevel features integration.
- We introduce the dual-stream gaze pyramid transformer (GazeDPTR). Employing perspective transformation, we rotate virtual cameras to normalize images, utilizing camera pose to merge normalized and original images for accurate gaze estimation. 

Please visit our <a href='https://yihua.zone/work/ivgaze/'>project page </a> for details. The dataset is available on <a href='https://github.com/yihuacheng/IVGaze/blob/main/DATASET.md'> this page </a>.

## Requirement

1. Install Pytorch and torchvision. This code is written in `Python 3.8` and utilizes `PyTorch 1.13.1` with `CUDA 11.6` on Nvidia GeForce RTX 3090. While this environment is recommended, it is not mandatory. Feel free to run the code on your preferred environment.

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

2. Install other packages.
```
pip install opencv-python PyYAML easydict warmup_scheduler
```

If you have any issues due to missing packages, please report them. I will update the requirements. Thank you for your cooperation.

## Training

**Step 1: Choose the model file.** 

We provide three models `GazePTR.py`, `GazeDPTR.py` and `GazeDPTR_v2.py`. The links of pre-trained weights are the same. Please load corresponding weights based on your requirement.

| | Name | Description | Input | Output|Accuracy|Pretrained Weights|
|:----|:----|:----|:----:|:----:|:----:|:----:|
|1|GazePTR| This method leverages multi-level feature.|Normalized Images|Gaze Directions|7.04°|<a href='https://drive.google.com/file/d/1uc5OkwZJO-KfMSuFAW0Hl-JSwTrfPURq/view?usp=drive_link'> Link </a>|
|2|GazeDPTR| This method integrates feature from two images.|Normalized Images  Original Images|Gaze Directions|6.71°|<a href='https://drive.google.com/file/d/1uc5OkwZJO-KfMSuFAW0Hl-JSwTrfPURq/view?usp=drive_link'> Link </a>|
|3|GazeDPTR_V2| This method contains a diffierential projection for gaze zone prediction. |Normalized Images  Original Images|Gaze Directions Gaze Zone|6.71° 81.8%|<a href='https://drive.google.com/file/d/1uc5OkwZJO-KfMSuFAW0Hl-JSwTrfPURq/view?usp=drive_link'> Link </a>|

Please choose one model and rename it as `model.py`, *e.g.*,
```
cp GazeDPTR.py model.py
```

**Step 2: Modify the config file**


Please modify `config/train/config_iv.yaml` according to your environment settings.

- The `Save` attribute specifies the save path, where the model will be stored at`os.path.join({save.metapath}, {save.folder})`. Each saved model will be named as `Iter_{epoch}_{save.model_name}.pt`
- The `data` attribute indicates the dataset path. Update the `image` and `label` to match your dataset location.

**Step 3: Training models**

Run the following command to initiate training. The argument `3` indicates that it will automatically perform three-fold cross-validation:

```
python trainer/leave.py config/train/config_iv.yaml 3
```

Once the training is complete, you will find the weights saved at `os.path.join({save.metapath}, {save.folder})`. 
Within the `checkpoint` directory, you will find three folders named `train1.txt`, `train2.txt`, and `train3.txt`, corresponding to the three-fold cross-validation. Each folder contains the respective trained model."

## Testing
Run the following command for testing.
```
python tester/leave.py config/train/config_iv.yaml config/test/config_iv.yaml 3
```
Similarly, 
- Update the `image` and `label` in `config/test/config_iv.yaml` based on your dataset location.
- The `savename` attribute specifies the folder to save prediction results, which will be stored at `os.path.join({save.metapath}, {save.folder})` as defined in `config/train/config_iv.yaml`.
- The code `tester/leave.py` provides the gaze zone prediction results. Remove it if you do not require gaze zone prediction.

## Evaluation

We provide `evaluation.py` script to assess the accuracy of gaze direction estimation. Run the following command:
```
python evaluation.py {PATH}
```
Replace `{PATH}` with the path of `{savename}` as configured in your settings.

Please find the visualization code in the issues.

## Contact
Please send email to `y.cheng.2@bham.ac.uk` if you have any questions. 
