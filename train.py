# DS_SKIP_CUDA_CHECK=1
# from cldm.hack import disable_verbosity, enable_sliced_attention
# disable_verbosity()
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from fivek_dataset import create_webdataset
import webdataset as wds
from pytorch_lightning.callbacks import ModelCheckpoint
from PE_Net import PE_Net
# Configs

resume_path = './checkpoints/fivek10k.ckpt'
fivek_images = './data/FiveK_C/train'
# coco_images = './COCO-2017/*/*.*'

# The actual batch size is batch_size * number_of_gpu
batch_size = 1
number_of_gpu = 1
learning_rate = 1e-4
logger_freq = 1000
name = f"FiveK"

sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
state_dict = load_state_dict('./models/control_sd15_ini.ckpt', location='cpu')
new_state_dict = {}
for s in state_dict:
    if "cond_stage_model.transformer" not in s:
        new_state_dict[s] = state_dict[s]
model.load_state_dict(new_state_dict)

ckpt = load_state_dict('./checkpoints/fivek10k.ckpt', location='cpu')

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = create_webdataset(
    data_dir=fivek_images,
)

dataloader = wds.WebLoader(
    dataset          =   dataset,
    batch_size       =   batch_size,
    num_workers      =   0,
    pin_memory       =   False,
    prefetch_factor  =   None,
)

logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
    dirpath                   =     'checkpoints',
    filename                  =     name + '-{epoch:02d}-{step}',
    monitor                   =     'step',
    save_last                 =     False,
    save_top_k                =     -1,
    verbose                   =     True,
    every_n_train_steps       =     10000,   # How frequent to save checkpoint
    save_on_train_epoch_end   =     True,
)


trainer = pl.Trainer(devices                   =     number_of_gpu,
                     precision                 =     32,
                     sync_batchnorm            =     True,
                     accelerator               =     'gpu',
                     callbacks                 =     [logger, checkpoint_callback])

# Train
if __name__ == '__main__':
    # trainer.fit(model, dataloader)
    trainer.fit(model, dataloader, ckpt_path=ckpt)

# If you want to continue training from a pytorch-lightning checkpoint, you can use
# trainer.fit(model, dataloader, ckpt_path="XXXX.ckpt")