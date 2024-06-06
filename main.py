#_# Python Libraries 
import os
import argparse
import numpy as np
import torch
import torchvision
from torchinfo import summary
import torch.optim as optim
import torch.nn as nn

#_# Our Source Code
from src import datasets, utils
from src.models import simple_cnn, wide_resnet, transformer
import exp_setup

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 导入并使用预训练的Vision Transformer模型：
from transformers import ViTForImageClassification, ViTImageProcessor

from PIL import Image

# 在处理CIFAR-10数据时，原始图像尺寸为32x32，而ViT模型通常需要224x224的输入图像。请确保在数据预处理中调整图像大小：
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Vision Transformer模型需要特定的输入格式和预处理，使用ViTFeatureExtractor来进行预处理：
def preprocess_data_batch(data, labels, transform, batch_size=32):
    processed_data = []
    processed_labels = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_data = [transform(Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))) for image in batch_data]
        batch_data = torch.stack(batch_data)
        processed_data.append(batch_data)
        processed_labels.append(torch.tensor(batch_labels))
    return torch.cat(processed_data), torch.cat(processed_labels)


if __name__ == "__main__":

    args = exp_setup.get_parser().parse_args()

    args.device = "cpu"
    if torch.cuda.is_available():
        args.device = "cuda"        
        torch.cuda.set_device(args.gpu_id)
    print(torch.cuda.get_device_properties(args.device))
    
    if args.reproducibility:
        torch.manual_seed(args.rand_seed)
        torch.cuda.manual_seed(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)

    if args.split == 0:
        train_data, train_labels,  test_data,  test_labels = datasets.get_dataset(args, verbose=1)

        
        # 使用预训练的Vision Transformer：
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        train_data, train_labels = preprocess_data_batch(train_data, train_labels, transform)
        test_data, test_labels = preprocess_data_batch(test_data, test_labels, transform)
  
        train_data = processor(images=train_data, return_tensors="pt")['pixel_values']
        test_data = processor(images=test_data, return_tensors="pt")['pixel_values']
    
        
        dataset = ((train_data, train_labels),(None,None), (test_data,  test_labels)) 

    else:
        raise Exception("Setting of split == 1 is not implemented in this version")    
        # train_data, train_labels, valid_data, valid_labels, test_data,  test_labels = datasets.get_dataset(args, verbose=1)
        # dataset = ((train_data, train_labels), (valid_data, valid_labels), (test_data,  test_labels))  
    

    # 使用预训练的ViT模型
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=args.num_classes).to(args.device)
    
    # For Transformer
    #model = transformer.TransformerWithSalt(input_channels=args.num_input_channels,  # 使用图像数据相关的参数
    #                            model_dim=args.model_dim,
    #                            num_heads=args.num_heads,
    #                            num_layers=args.num_layers,
    #                            num_classes=args.num_classes,
    #                            salt_layer=args.salt_layer).to(args.device)

    # For LeNet
    #model = simple_cnn.SimpleCNN(num_classes=args.num_classes, salt_layer=args.salt_layer,
    #                            mean =  datasets.CIFAR10_MEAN, 
    #                            std = datasets.CIFAR10_STD, 
    #                            num_input_channels=args.num_input_channels)

    ## For WideResNet
    # model = wide_resnet.WideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)
    ## For SaltedWideResNet
    # model = wide_resnet.SaltyWideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)

    ## For ConvNet (PAMAP2)
    # model = simple_cnn.SenNet(salt_layer=args.salt_layer)
        
    model.to(args.device)
    # 针对Resnet、Transformer以下内容需要注释掉
    #if args.dataset == "cifar10":
    #    if args.salt_layer == -1:
    #        summary(model, [(1, args.num_input_channels, 32, 32)], device=args.device)       
    #    elif 0<= args.salt_layer <=5: 
    #        summary(model, [(1, args.num_input_channels, 32, 32),(1,1,1,1)], device=args.device)       
    #    else:
    #        summary(model, [(1, args.num_input_channels, 32, 32),(1,args.num_classes)], device=args.device)       
    #elif args.dataset == "pamap":
    #    summary(model, [(1, 1, 27, 200),(1,1,1,1)], device=args.device)    
    
    # 针对Transformer以下内容需要打开：
    #summary(model, input_size=(args.batch_size, args.num_input_channels, 32, 32), device=args.device)
    # 针对预训练的ViT模型，以下内容需要打开：
    summary(model, input_size=(args.batch_size, 3, 224, 224), device=args.device)
    
    utils.train_test(args, model, dataset, save_model=True)
