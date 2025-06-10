import wandb

def setup_wandb(run_name, hyperparams, project_name, entity):
    try:
        wandb.init(project=project_name, entity=entity, config=hyperparams)
        wandb.run.name = run_name
    except:
        raise ValueError("Could not initialize wandb. Please check your settings.")

def log_wandb(history, mode="train"):
    if mode == "eval":
        wandb.run.summary.update(history)
    else:
        wandb.log(history)

import torch
import torch.nn.init as init
def custom_weight_init(m, init_option='kaiming_normal'):
    if isinstance(m, torch.nn.Linear):
        if init_option == 'xavier_uniform':
            init.xavier_uniform_(m.weight)
        elif init_option == 'xavier_normal':
            init.xavier_normal_(m.weight)
        elif init_option == 'uniform':
            init.uniform_(m.weight, -0.9, 0.9)
        elif init_option == 'normal':
            init.normal_(m.weight, 0, 0.1)
        elif init_option == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight)
        elif init_option == 'kaiming_normal':
            init.kaiming_normal_(m.weight)
        else:
            raise ValueError('Invalid weight initialization scheme')
        
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Sequential):
        for sub_m in m:
            custom_weight_init(sub_m, init_option)
    elif isinstance(m, torch.nn.Parameter):
        if init_option == 'xavier_uniform':
            init.xavier_uniform_(m)
        elif init_option == 'xavier_normal':
            init.xavier_normal_(m)
        elif init_option == 'uniform':
            init.uniform_(m, -0.9, 0.9)
        elif init_option == 'normal':
            init.normal_(m, 0, 0.1)
        elif init_option == 'kaiming_uniform':
            init.kaiming_uniform_(m)
        elif init_option == 'kaiming_normal':
            init.kaiming_normal_(m)
        else:
            raise ValueError('Invalid weight initialization scheme')