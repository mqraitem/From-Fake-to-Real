# From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Spurious Correlations in Image Recognition

Official Pytorch implementation of [From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Spurious Correlations in Image Recognition](https://arxiv.org/abs/2308.04553). 

## Setup

### Set up conda environment  
```
conda env create -f ffr_env.yml
```

### Prepare dataset.

- CelebA  
Download [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) 

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) 

- SpuCo Animals  
Download [SpuCo Animals](https://github.com/BigML-CS-UCLA/SpuCo) 

Please then edit the dir of of each dataset in dataset_dirs.txt. The information required to reproduce the the splits are in data/. The code will automatically load the right splits. 

## Image Generation

To generate the synthetic data required for training on:

#### Utk-Face

```
python generation_code/generate_images_utk_face.py --gender [Male/Female] --bias [old/young]
```

Make sure to run the script for all the combinations: Male-old/Male-young/Female-old/Female-young

#### CelebA-HQ

```
python generation_code/generate_images_celeba.py --gender [Male/Female] --attr [Smiling] --class_idx [0/1]
```

Make sure to run the script for all the combinations: Male-0/Male-1/Female-0/Female-1


#### SpuCO Animals 

```
python generation_code/generate_images_celeba.py --gender [big_dog/small_dog/landbird/waterbird] --bias [Indoors/Outdoors/Land/Water]
```

Make sure to run the script for all the combinations: big_dog-Indoors/big_dog-Outdoors/small_dog-Indoors/small_dog-Outdoors/landbird-Land/landbird-Water/waterbird-Land/waterbird-Water


## Training


The training scripts could be found under training_code/. Each script is named following the format: 

```
train_[TRAINING_METHOD]_[SYNTHETIC_AUG_METHOD].py
```

where TRAINING_METHOD could be: erm/resample/groupdro/dfr and SYNTHETIC_AUG_METHOD could be: none/usb/asb/ffr_1/ffr_2. Note that ffr_1 refers to step 1 of FFR, and ffr_2 refers to step 2 of FFR. Refer to the paper for further info. 


To run any of the methods, simply run: 

```
python training_code/train_[TRAINING_METHOD]_[SYNTHETIC_AUG_METHOD].py --dataset [DATASET] --minority_to_keep [BIAS_RATE] --seed [SEED] 
```

where DATASET: celeba/utk_face/spuco_animals. and BIAS_RATE: [0.9/0.95/0.97/0.99/0.999] except for SpuCO Animals where minimum is 0.95. 



## Citation 

If you find this repository useful please give it a star and cite as follows! :) :
```
    @article{qraitem2023fake,
    title={From Fake to Real (FFR): A two-stage training pipeline for mitigating spurious correlations with synthetic data},
    author={Qraitem, Maan and Saenko, Kate and Plummer, Bryan A},
    journal={arXiv preprint arXiv:2308.04553},
    year={2023}
    }
```