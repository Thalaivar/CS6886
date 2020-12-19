import yaml

from models.DSNAS.network_eval import ShuffleNetV2_OneShot

def load():
    with open('./models/DSNAS/DSNASsearch240.yaml') as f:
        config = yaml.load(f)
    
    args = {}
    for key in config:
        for k, v in config[key].items():
            args[k] = v

    args['loc_mean'] = 1
    args['loc_std'] = 0.01

    scale_list = 8*[1.0]
    scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    channels_scales = []
    for i in range(len(scale_ids)):
        channels_scales.append(scale_list[scale_ids[i]])

    model = ShuffleNetV2_OneShot(args=args, architecture=args['arch'], channels_scales=channels_scales)
    return model