## [ECCV 2022] D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution

<h4 align="center">Youwei Li$^1$, Haibin Huang$^2$, Lanpeng Jia$^1$, Haoqiang Fan$^1$, Shuaicheng Liu$^{3,1}$</center>
<h4 align="center">1. Megvii Research, 2. Kuaishou Technology</center>
<h4 align="center">3. University of Electronic Science and Technology of China</center><br><br>





#### This is the official MegEngine implementation of our ECCV2022 paper "[***D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution***](https://arxiv.org/abs/2103.14373)".

Welcome to try megengine to train your model，and our PyTorch version will also be coming soon at [D2C-SR-PyTorch](https://github.com/Well-Lee-pro/D2C-SR).


## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/180904384-fefbaf33-feac-45ad-927e-da87e5d046f1.JPG)
Two stages in D2C architecture: (a) Divergence stage, (b) Convergence stage. Divergence network with tree-based structure outputs multi-predictions with different high-frequency recovery. Convergence network obtains more accurate result by weighted combining divergence results. (c) Divergence loss.


### Dependencies

* MegEngine>=1.3.1
* tqdm

### Data Preparation

#### RealSR and DRealSR
RealSR and DRealSR has been made public and can be downloaded from their official repo.

#### D2CRealSR
Our D2CRealSR dataset can be download from [Google Drive](https://drive.google.com/file/d/1ZTjB6q94ge2h9ixf1osEGXXnfuLTYVzO/view?usp=sharing).

#### Make Dataset List
Before training and validation, a data list in ```.txt``` format needs to be prepared, and input the path of your data list during training and evaluation. 
You need to prepare data lists for different scale factors as well as for training and validation datasets.

The data list has the following format：

```
absolute_path_LR absolute_path_HR

——————————————————————————————————————————————————————

For example (RealSR)

    List of x4 scale:
    
        ——————————————————————————————————————————————
        /data/Canon_034_LR4.png /data/Canon_034_HR.png
        /data/Canon_035_LR4.png /data/Canon_035_HR.png
        ...
    
        ——————————————————————————————————————————————
    
    List of x2 scale:
        ——————————————————————————————————————————————
        /data/Canon_034_LR2.png /data/Canon_034_HR.png
        /data/Canon_035_LR2.png /data/Canon_035_HR.png
        ...
        
        ——————————————————————————————————————————————

```


### Training

To train the model, you can run:

```
python train.py --train_list_path your_train_list_path --val_list_path your_val_list_path --scale 4 --ex_id your_exp_name
```

### Evaluation

For evaluation, load the pretrained checkpoint and run:

```
python test.py --checkpoint checkpoint_path --val_list_path your_val_list_path
```

MegEngine checkpoint can be download from [Google Drive](https://drive.google.com/file/d/1itbkFWQ8ZP9F9XcDYpac16J6vspJ2wiV/view?usp=sharing).



### Citations

To be updating...

### Contact

Contact Email: [liyouwei.wellee@gmail.com](liyouwei.wellee@gmail.com)

### Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [RCAN](https://github.com/yulunzhang/RCAN)
* [CDC](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)

We thank the respective authors for open sourcing their methods.
