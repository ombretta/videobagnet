import torch
from torch import nn
import torchvision.models as models

from models import (vid_bagnet_tem, resnet, mlp)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []

    add_flag = False
    for k, v in model.named_parameters():
        #print("model parameter", get_module_name(k))
        if ft_begin_module == get_module_name(k):
            add_flag = True
            #print("Start fine-tuning from here.")

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'vidbagnet', 'vidbagnet_tem', 'resnet', 'resnet2p1d', 'preresnet', 
        'wideresnet', 'resnext', 'densenet', 'I3D', 'I3D_segments', 'I3D_1d',
        '1d_cnn', '1d_cnn_wide_to_narrow', '1d_cnn_shallow', '1d_cnn_shallow_narrow', 'MLP'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    if opt.model == 'vidbagnet_tem':
        model = vid_bagnet_tem.generate_model(model_depth=opt.model_depth,
                                      receptive_size=opt.receptive_size,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'MLP':
        model = mlp.generate_model(opt.n_classes, n_input_channels=opt.n_input_channels,
                                              multilabel=opt.multilabel)

    
    return model


def adjust_weights_dimensions(model_dict, pretrained_dict, ):
    '''When the model has larger weights than the checkpoint, adjust by:
        - replicate channels: replicate weights until the dimensions match
        - '''
    
    adjusted_pretrained_dict = pretrained_dict.copy()
    
    for weight in adjusted_pretrained_dict:
        
        weight_tensor = adjusted_pretrained_dict[weight]
        dim_pretrained = pretrained_dict[weight].shape
        dim_model = model_dict[weight].shape
        
        if dim_pretrained != dim_model:
        
            ''' Pretrained weights dimensions > Model weights dimensions '''
            
            adj_dims = [min(p,m) for p, m in zip(dim_pretrained, dim_model)]
            
            ''' Take the middle temporal kernel
            note: loading the central tensor kernel (see paper In Defense 
            of Image Pre-Training for Spatiotemporal Recognition)'''
                
            ''' Take the the average in the temporal dimension. '''
            
            if len(dim_model)==5:
                if dim_model[2] == 1:
                    weight_tensor = torch.mean(weight_tensor, dim=2, keepdim=True)
                weight_tensor = weight_tensor[:adj_dims[0], 
                      :adj_dims[1], :adj_dims[2], :adj_dims[3], :adj_dims[4]] if \
                    len(dim_model)==5 else weight_tensor[:adj_dims[0]]
            else: weight_tensor[:adj_dims[0]]
            
            ''' Pretrained weights dimensions < Model weights dimensions '''
                
            to_add = [m-p if m>p else 0 for p, m in zip(weight_tensor.shape, dim_model)]
            
            if len(dim_model) != 5:
                   tensor_to_add = weight_tensor[:to_add[0]]
                   weight_tensor = torch.cat((weight_tensor, tensor_to_add))
            else:
                # channel in 
                tensor_to_add = weight_tensor[:to_add[0], :, :, :, :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 0)
                # channel out
                tensor_to_add = weight_tensor[:, :to_add[1], :, :, :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 1)
                # time 
                tensor_to_add = weight_tensor[:, :, :to_add[2], :, :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 2)
            
            adjusted_pretrained_dict[weight] = weight_tensor
            
    return adjusted_pretrained_dict



def inflate_2d_weights(model_dict):
    
    resnet50 = models.resnet50(pretrained=True)
    imagenet_dict = resnet50.state_dict()
    
    adjusted_pretrained_dict = {}
     
    for weight in [w for w in imagenet_dict if w not in ["fc.weight", "fc.bias"]]:
        
        weight_tensor = imagenet_dict[weight]
        
        dim_pretrained = weight_tensor.shape
        dim_model = model_dict[weight].shape
        
        if len(dim_pretrained) > 0:
            # channel in 
            tensor_to_add = weight_tensor[:dim_model[0]-dim_pretrained[0]]
            weight_tensor = torch.cat((weight_tensor, tensor_to_add), 0)
            
        if len(dim_pretrained) == 4:
            
            # Adds time dimension to weights
            weight_tensor = weight_tensor[:, :, None, :, :]
            
            diff = [m-p if m>p else 0 for p, m in zip(
                weight_tensor.shape, dim_model)]
            
            while any(torch.tensor(diff)>0):
                
                # channel out
                tensor_to_add = weight_tensor[:, :diff[1], :, :, :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 1)
                
                # time 
                tensor_to_add = weight_tensor[:, :, :diff[2], :, :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 2)
                
                # height 
                tensor_to_add = weight_tensor[:, :, :, :diff[3], :]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 3)
                
                # width 
                tensor_to_add = weight_tensor[:, :, :, :, :diff[4]]
                weight_tensor = torch.cat((weight_tensor, tensor_to_add), 4)
                
                diff = [m-p if m>p else 0 for p, m in zip(
                    weight_tensor.shape, dim_model)]
                
        adjusted_pretrained_dict[weight] = weight_tensor            
        
        # if adjusted_pretrained_dict[weight].shape != dim_model:
            # print(weight, adjusted_pretrained_dict[weight].shape, dim_model)
            
    return adjusted_pretrained_dict



def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes, device):
    ''' Ombretta: modified to load weights in video bagnet from 2d and 
    3d resnet checkpoints. '''
    
    if not pretrain_path: return model
    
    print('loading pretrained model {}'.format(pretrain_path))
    
    model_dict = model.state_dict()
    
    if 'ImageNet' in pretrain_path.stem: 
        print("imagenet!")
        ''' Here we are loading 2D ResNet ImageNet weights.'''
        adjusted_pretrained_dict = inflate_2d_weights(model_dict)
    
    else:
        ''' Here we are loading 3D ResNet Kinetics weights. '''
        pretrain = torch.load(pretrain_path, map_location=device)
        if 'state_dict' in pretrain:
            pretrained_dict = pretrain['state_dict']
        else: pretrained_dict = pretrain
        
        classification_layer = "conv3d_0c_1x1.conv3d" if "I3D" in model_name else "fc"
        n_pretrain_classes = pretrained_dict[classification_layer+".bias"].shape[0]
        print("n_pretrain_classes", n_pretrain_classes)
        print("n_finetune_classes", n_finetune_classes)
        
        # 1. filter out unnecessary keys and the final classification layer 
        # if the number of classes is different
        if n_pretrain_classes != n_finetune_classes:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in 
                           model_dict and k not in ["fc.weight", "fc.bias", 
                                                "conv3d_0c_1x1.conv3d.weight", 
                                                "conv3d_0c_1x1.conv3d.bias"]}
        
        # 1.1. filter out unnecessary keys 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in 
                            model_dict}
        
        # 1.5 Adjust filters to match the model filter sizes and number of channels
        adjusted_pretrained_dict = adjust_weights_dimensions(
            model_dict, pretrained_dict)
    
    # 2. overwrite entries in the existing state dict
    model_dict.update(adjusted_pretrained_dict) 

    #print("model_dict", {k:model_dict[k] for k in model_dict if 'conv3d_0c_1x1.conv3d'})    

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # print("model", model)
    
    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        print("Using multiple GPUs")
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
