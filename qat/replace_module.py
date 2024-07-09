import torch.nn as nn
from .learnable_binarizer import BinaryLinearWscales
from .base_binarizer import BinaryInterface

def replace_with_learnable_binarylinear(model, scaling_pattern, keep_parts):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if any(kp in name for kp in keep_parts):
            continue
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            qlinear = BinaryLinearWscales(
                module.weight, module.bias, scaling_pattern
            )
            print(name, module.weight.shape, qlinear.weight.shape)
            setattr(father, name[ind + 1 :], qlinear)

    return model


def binarylinear_to_regularlinear(model):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, BinaryInterface):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            linear = module.to_regular_linear()
            setattr(father, name[ind + 1 :], linear)
            print(f"replace layer {name} with {linear}")

    return model


def check_para_state(model):
    trainable_params, all_param = 0, 0
    for key, param in model.named_parameters():
        trainable = 'T*'
        all_param += param.numel()
        if param.requires_grad:
            trainable = 'T+'
            trainable_params += param.numel()
        else:
            trainable = 'T-'
        print(trainable, key)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    for name, module in model.named_modules():
        binarized = 'B*'
        if isinstance(module, nn.Linear):
            binarized = 'B-'
        elif isinstance(module, BinaryInterface):
            binarized = 'B+'
        print(binarized, name)