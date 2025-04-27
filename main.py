import argparse
import json
import os
import torch
from dataset.data import AudioDataset, get_dataloader
from run.train import do_train


def run(args):
    # all_path_arg = args.runner_path
    all_path_arg = "usage/runner.json"
    f_paths = open(all_path_arg, "r")
    paths_config = json.load(f_paths)

    input_path_arg = paths_config["input_path_arg"]
    f_input = open(input_path_arg, "r")
    input_config = json.load(f_input)

    output_path_arg = paths_config["output_path_arg"]
    f_output = open(output_path_arg, "r")
    output_config = json.load(f_output)

    data_config_path_arg = paths_config["data_config_path_arg"]
    f_data_config = open(data_config_path_arg, "r")
    data_config = json.load(f_data_config)

    model_config_path_arg = paths_config["model_config_path_arg"]
    f_model_config = open(model_config_path_arg, "r")
    model_config = json.load(f_model_config)

    run_config_path_arg = paths_config["run_config_path_arg"]
    f_run_config = open(run_config_path_arg, "r")
    run_config = json.load(f_run_config)

    do_save = args.do_save
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_config["output_dir"]):
        os.makedirs(output_config["output_dir"])

    data_dir = input_config["data_dir"]
    batch_size = data_config["batch_size"]
    shuffle = data_config["do_shuffle"]

    audio_dataset = AudioDataset(data_path=data_dir,
                                 data_config=data_config,
                                 num_points=20)

    audio_dataloader = get_dataloader(audio_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle)

    do_train(dataloader=audio_dataloader,
             run_config=run_config,
             data_config=data_config,
             model_config=model_config,
             output_config=output_config,
             do_save=do_save)


class Arguments:
    def __init__(self, runner_path, do_save, do_parse=True):
        self.do_parse = do_parse
        self.runner_path = runner_path
        self.do_save = do_save


if __name__ == "__main__":
    print("Running the system..")
    # parser = argparse.ArgumentParser(description='Running the system')
    # parser.add_argument("--do_parse", type=int, required=True)
    # parser.add_argument("--runner_path", type=str)
    # parser.add_argument("--do_save", type=int)
    # args = parser.parse_args()
    # if args.do_parse:
    #     run(args)
    # else:
    #     print("Please run this system through the terminal with required configuration file")

    args = Arguments(runner_path="usage/runner.json", do_save=True)
    run(args)
