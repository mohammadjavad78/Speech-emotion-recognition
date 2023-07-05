
import yaml

# Load config file
def Getyaml(filename='config.yml'):
    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_dict={}

    # Access hyperparameters
    config_dict['model_name'] = config['model_name']
    config_dict['ckpt_save_path'] = config['ckpt_save_path']
    config_dict['ckpt_path'] = config['ckpt_path']
    config_dict['report_path'] = config['report_path']
    config_dict['learning_rate'] = config['learning_rate']
    config_dict['batch_size'] = config['batch_size']
    config_dict['num_epochs'] = config['num_epochs']
    config_dict['gamma'] = config['gamma']
    config_dict['step_size'] = config['step_size']
    config_dict['ckpt_save_freq'] = config['ckpt_save_freq']
    config_dict['dataset'] = config['dataset']['path']
    config_dict['meanonnotzero'] = config['meanonnotzero']

    return config_dict