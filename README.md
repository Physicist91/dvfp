# Deep Learning for Fine-grained Visual Classification

Here we implement state-of-the-art techniques for fine-grained classification.

1. [Discriminative filter learning within CNN](https://arxiv.org/abs/1611.09932) -- CVPR, June 2018
2. [Bilinear CNN](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) -- ICCV, 2015

To this end, we authored a few notebooks that illustrate both techniques as well as other approaches:
- `Plain ResNet50, VGG19_bn.ipynb` contain codes and description for our fine-tuned ResNet-50 and VGG-19 models.
- `dfl.ipynb` outlines major components and implementation of the DFL architecture.
- `BiLinPol FastAI CARSET.ipynb` and `BilinearModel.ipynb` contain a simple implementation of Bilinear CNN
- `ResNet34.ipynb` contains a straightforward transfer learning approach using ResNet34.

Datasets:
+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
+ [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

<tr>
        <td>
          <table width="800" cellpadding="0" cellspacing="0">
            <tbody><tr>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car1.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car2.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car3.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car4.jpg" width="200" height="140"></td>
            </tr>
            <tr>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car5.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car6.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car7.jpg" width="200" height="140"></td>
              <td><img src="https://ai.stanford.edu/~jkrause/cars/car8.jpg" width="200" height="140"></td>
            </tr>
          </tbody></table>
        </td>
</tr>

Implemented using [PyTorch](https://pytorch.org/). This work is done as part of the final project for **Deep Vision, ST2019**.

This work is also partly inspired by
- [Weakly supervised Data Augmentation Network](https://arxiv.org/abs/1901.09891), ArXiV preprint, March 2019.

Usage:
```
# clone this repo
git clone https://github.com/Physicist91/dvfp

# first edit the parameters (e.g. GPU number, batch_size) then execute run.sh
# to resume training, specify the path to weights in --resume
./run.sh
```

Final results

![table](results.png)

The weights can be downloaded from here. The poster for this project can be found below (note that the numbers/results have been updated -- refer to the table above).

*****

![poster](poster.png)
