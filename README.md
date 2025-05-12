# EfficientViT
Unofficial c++ LibTorch implementation of EfficientViT

EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction.

Original PyTorch code: [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)

Original EfficientViT paper: [arxiv.org/2205.14756](https://arxiv.org/abs/2205.14756)

#### Dependencies 
- [libTorch](https://pytorch.org)

- [OpenCV](https://opencv.org/releases/) 

- Thanks to Carson Poole for this great repository [safetensors](https://github.com/carsonpo/safetensors.cpp)

#### Model
Download from [HuggingFace](https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0/tree/main)

#### Run

    EfficientViT.exe --img=imgs\IMG_6265.JPG --vit=data\dc-ae-f32c32-mix-1.0\model.safetensors --name=dc-ae-f32c32-in-1.0 --show=1

#### DCAE example
![first_result](https://github.com/user-attachments/assets/3b04ff51-e071-418c-acfc-0c2f503c2138)






