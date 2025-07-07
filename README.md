# Robust Zero-Shot Crowd Counting and Localization With Adaptive Resolution SAM

## installation 
```
1. pip install -r requirements.txt
2. git clone https://github.com/facebookresearch/detectron2.git
3. python -m pip install -e detectron2
4. conda install mpi4py
5. pip install git+https://github.com/openai/whisper.git
```

## pretrained models
```
1. SEEM checkpoint: please download from [huggingface](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt)
2. CLIP checkpoint: please refer to [huggingface](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)
```

## Train and Test

```
1. generate segmentations for each image with SEEM as pseudo labels (run_gen_sam_labels.py)
2. train a counting network with pseudo labels (run_train_counter.py)
3. predict the location with the counting network (run_resam.py)
4. use the prediction as prompts to generate new masks with SEEM + point prompt (run_resam.py)
```

### Citation
If you use our code or models in your research, please cite with:

```
@inproceedings{wan2025robust,
  title={Robust Zero-Shot Crowd Counting and Localization With Adaptive Resolution SAM},
  author={Wan, Jia and Wu, Qiangqiang and Lin, Wei and Chan, Antoni},
  booktitle={European Conference on Computer Vision},
  pages={478--495},
  year={2025},
  organization={Springer}
}
```

