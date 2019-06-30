## SETUP
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.metrics import error_rate
import torch
print('pytorch version: ',torch.__version__)
import fastai
print('fastai version: ',fastai.__version__)
!pip freeze > './requirements.txt'

img_dir='data/car_data/train'
path=Path(img_dir)
data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2, #percentage to use for the validation set
                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=90,max_lighting=0.3),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)
print(np.random.choice(data.classes, 10))
#data.show_batch(rows=4, figsize=(40,40))

learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="model/")
learn.model

# will train total of 35 cycles
learn.fit_one_cycle(15)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interPlot = interp.plot_top_losses(9, figsize=(30,25))
interPlot.savefig('figures/interpolation_stage1.pdf', transparent=False)
interp.most_confused(min_val=2)

learn.unfreeze()
learn.fit_one_cycle(15)
learn.lr_find()
lr_plot = learn.recorder.plot()
lr_plot.savefig('figures/lr_plot_stage1.pdf', transparent = False)

learn.unfreeze() 
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(30,25))
interPlot = interp.plot_top_losses(9, figsize=(30,25))
interPlot.savefig('figures/interpolation_stage2.pdf', transparent=False)