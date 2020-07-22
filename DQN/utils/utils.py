import os
import torch


def save_tb_scalars(writer, epoch, **kwargs):
    summary_string = f"{epoch:4}"
    for metric in kwargs:
        summary_string += f"  |  {metric} {kwargs[metric]:10.3}"
        writer.add_scalar(f'rl/{metric}', torch.tensor(kwargs[metric], dtype=torch.float32), epoch)
    print(summary_string)


def unique_path(path):
    """

    :str path: path which should not be overridden
    """
    if os.path.exists(path):
        expand = 1
        not_finished = True
        while not_finished:
            if os.path.exists(path + "_{}".format(expand)):
                expand += 1
            else:
                return path + "_{}".format(expand)
    return path
