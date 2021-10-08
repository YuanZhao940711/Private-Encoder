# Private-Encoder: Enforcing Privacy in Latent Space for Human Face Images

<p align="center">
<img src="docs/Comparison_Images.png" width="800px"/>
</p>

## Description
Official pytorch impementation of our private-encoder paper for both training and inferencing.

## Usage

### Preparation
- Install required packages
``` 
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```
- Download required pretrained models

The auxiliary models which are needed for training your own model from scratch or inference.

| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained Arcface for use in our ID loss during training.
|[MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view?usp=sharing)  | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)

Please download and save these models to the directory `pretrained_models`.

### Training

- The main training script can be found in `scripts/train.py`.
- Before training, please define your data path at `--train_source_root` and `--test_source_root`.
- The training results are saved to `exp_dir=./experiment`, so you can check the checkpoints, tensorboard logs, intermediate train output and test outputs in that directory.
- To acquire different anonymization effects, you can adjust `--target_id`.

```
python scripts/train.py \
--exp_dir=./experiment \
--dataset_type=celeba_encode \
--checkpoint_path=./pretrained_models/psp_ffhq_encode.pt \
--train_source_root=/content/drive/MyDrive/Image_datasets/celeba_align_20k_21k_aligned \
--test_source_root=/content/drive/MyDrive/Image_datasets/celeba_align_20k_21k_aligned \
--workers=8 \
--batch_size=4 \
--test_workers=8 \
--test_batch_size=4 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--id_lambda=0.1 \
--l2_lambda=1.0 \
--max_steps=100000 \
--image_interval=5000 \
--board_interval=50 \
--val_interval=1000 \
--save_interval=5000 \
--target_id=0.0
```

You can also directly run training colab notebook example in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11K12U9p0GRtPCI3pkcCgzA5PgJa7SyhV?usp=sharing)

### Inference

- Pretrained Models

Please download the pre-trained Private-Encoder models from the following links. 

| Path | Description
| :--- | :----------
|[Pretrained Models for FFHQ](https://drive.google.com/drive/folders/1HR0YLWakpnd1eQw_4YGTOelo_mzzuBAk?usp=sharing)  | trained with the FFHQ dataset for different level of identity anonymization.

- If you wish to use one of the pretrained models for training or inference, please change the directory of `--checkpoint_path`.
- Before inference, please define your data path at `--data_path`.
```
python scripts/inference.py \
--exp_dir=./image_datasets \
--checkpoint_path=./best_models/privacy_0.0/best_model.pt \
--data_path=./image_datasets/ffhq5k_6k_related/ffhq5k_6k_aligned \
--test_batch_size=5 \
--test_workers=4
```

You can also directly run inference colab notebook example in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1waOYFQ3Z-QcgMk4wz1c2041ManPSmVb0?usp=sharing)

## Citation
If you use this code for your research, please cite our paper Private-Encoder: Enforcing Privacy in Latent Space for Human Face Images:

```
@article{zhao2021private,
  title={Private-encoder: Enforcing privacy in latent space for human face images},
  author={Zhao, Yuan and Liu, Bo and Zhu, Tianqing and Ding, Ming and Zhou, Wanlei},
  journal={Concurrency and Computation: Practice and Experience},
  pages={e6548},
  year={2021},
  publisher={Wiley Online Library}
}
```
