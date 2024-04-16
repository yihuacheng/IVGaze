![](./assets/logo.png)
<h1> What Do You See in Vehicle? Comprehensive Vision Solution for In-Vehicle Gaze Estimation  <a href='https://arxiv.org/abs/2403.15664'><img src='https://img.shields.io/badge/ArXiv-PDF-red' style="vertical-align:middle;"></a>  </h1>

<i><a href="https://yihua.zone/">Yihua Cheng </a> and <a href='https://scholar.google.com.hk/citations?user=9ggbm0QAAAAJ&hl=en'>Feng Lu</a>,  ICCV 2023</i>
   
</div>
<br>

<img src="images/teaser.png" width = "400" height = "200" alt="图片名称" align=center />
</div>

## Description
Dual cameras have been applied in many devices recently. In this paper, we explore a new direction for gaze estimation. We propose a dual-view gaze estimation network (DV-Gaze) including dual-view interactive convolution block and dual-view transformers.
![DVGaze](images/pipeline.png)

## Usage
Please re-write the config file in `config/train/xxx.yaml` and `config/test/xxx.yaml`.

To train a model, Please run the command:
```
python trainer/total.py config/train/xxx.yaml
```

To evaluate a model, please run the command:
```
python tester/total.py config/train/xxx.yaml config/test/xxx.yaml
```

where `config/train/xxx.yaml` indicates the config for training model.


Please download the label file from <a href='https://drive.google.com/drive/folders/16yt3xjkQzR_hA5EMFWQhrL-s2f3A3MKb?usp=sharing'> Google Driver </a>. We are not authorized to distribute image files. Please access <a href='https://phi-ai.buaa.edu.cn/Gazehub/'> Gazehub </a> to acquire data-processing code on ETH and EVE. Note that the website only works on working time (8:00-19:00?) in the China time zone (I have graduated and don't know the reason). Apologise for the inconvenience. I will try my best to update data-processing codes in Github.

**All codes are beta version and we will keep updating this repositories. The full version will be updated before Oct. 10.**

Please send email to `y.cheng.2@bham.ac.uk` if you have any questions. 

*We can also meet in ICCV conference in person:) Have a good day.*
