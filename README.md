# Spatial Focused Bitemporal Interactive Network for Remote Sensing Images Change Detection

Here, we provide the pytorch implementation of the paper: Spatial Focused Bitemporal Interactive Network for Remote Sensing Images Change Detection.

![](./image/Network.png)


## Requirements

```txt
einops==0.7.0
grad_cam==1.4.8
matplotlib==3.6.0
numpy==1.23.5
opencv_python==4.5.4.58
Pillow==10.2.0
scipy==1.9.1
tifffile==2023.2.3
timm==0.4.12
torch==1.12.1
torchvision==0.13.1
tqdm==4.64.1
ttach==0.0.3
wandb==0.13.5
```

### Filetree

```
SFBI-Net
├─ README.md
├─ data_config.py
├─ datasets
│  ├─ CD_dataset.py
│  └─ data_utils.py
├─ eval.py
├─ image
│     ├─ Network.png
│     ├─ clcd.png
│     ├─ egy.png
│     └─ levir-cd.png
├─ misc
│  ├─ logger_tool.py
│  └─ metric_tool.py
├─ models
│  ├─ SFBI-Net.py
│  ├─ __init__.py
│  ├─ _utils.py
│  ├─ basic_model.py
│  ├─ evaluator.py
│  ├─ help_funcs.py
│  ├─ losses.py
│  ├─ networks.py
│  ├─ resnet.py
│  └─ trainer.py
├─ output
│  ├─ checkpoints
│  └─ vis
├─ requirements.txt
├─ script
│  ├─ eval_SFBI-Net.sh
│  └─ run_SFBI-Net.sh
├─ train.py
└─ utils.py

```

## Quick start

### Installation

clone this repo:

```sh
git clone https://github.com/Mryao-yuan/SFBI-Net.git
cd SFBI-Net
```

### Train

```sh
sh script/run_SFBI-Net.sh
``` 

### Test

```sh
sh script/eval_SFBI-Net.sh
``` 

### Qualitative Results

#### results on [LEVIR-CD](https://www.mdpi.com/2072-4292/12/10/1662/pdf)
![](./image/levir.png)

#### results on [CLCD](https://ieeexplore.ieee.org/abstract/document/10145434)
![](./image/clcd.png)

#### results on [EGY](https://ieeexplore.ieee.org/iel7/4609443/4609444/09780164.pdf)
![](./image/egy.png)

### Copyright

The project has been licensed by Apache-2.0. Please refer to for details. [LICENSE.txt](https://github.com/Mryao-yuan/SFBI-Net/LICENSE.txt)

### Thanks

* [Pytorch-Grad-Cam](https://github.com/jacobgil/pytorch-grad-cam)
* [BIT](https://github.com/justchenhao/BIT_CD)
* [ChangeFormer](https://github.com/wgcban/ChangeFormer)

(Our SFBI-Net is implemented on the code provided in this repository)

