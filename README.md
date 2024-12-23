# RetouchDiffusion 

This is the official PyTorch code for our paper Zero-Reference Low-Light Enhancement via Physical Quadruple Priors

![Framework](./assets/Framework.jpg)


#### 0. Preparation

Create a new conda environment
```
conda env create -f environment.yaml
conda activate Retouch
```

Download the checkpoints from Anonymous Google drive 

- `./savemodel/pe_net.pt` [link](https://drive.google.com/file/d/1YSNdR_nUxDFKZRMKz42UPpWYfG8C8qKp/view?usp=drive_link)
- `./checkpoints/Fivek.ckpt` [link](https://drive.google.com/file/d/1WVz8WF4OE-qlsVJ-FHxtkgRwmuGrBGfc/view?usp=drive_link)
- `./models/control_sd15_ini.ckpt` [link](https://drive.google.com/file/d/1XYbhNlWAJ3cRws2nNYHcCZJ8H8iEHUUc/view?usp=drive_link)

#### 1. Test

For testing the example images in `./test_data`, simply run:

```
CUDA_VISIBLE_DEVICES=0 python test.py 
```

Then the resulting images can be found in `./result`.


#### 2. Train

##### 2.1 Data Preparation
Our model is trained solely with the [MIT Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/).


##### 2.2 Train
Parameters can be edited in `train.py`, such as batch size (`batch_size`), number of GPUs (`number_of_gpu`), learning rate (`learning_rate`), how frequently to save visualization (`logger_freq`).

On NVIDIA GeForce RTX 3090, setting 2 batches per GPU takes 20GB memory for each GPU. We use 2 GPUs to train the framework.

If you want to train from scratch, please set `resume_path=''`. Currently it continues training from `checkpoints/FiveK.ckpt`.



-------

This code is based on [ControlNet](https://github.com/lllyasviel/ControlNet) 
