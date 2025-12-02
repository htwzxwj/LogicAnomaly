import logging
import os
import sys
import click
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # <repo_root>
sys.path.insert(0, str(ROOT))
import utils
import setproctitle
import src.StudentTeacher
from src.AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

setproctitle.setproctitle("LogicAnomaly-Training-StudentTeacher Model")

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
    "mvtecloco": ["datasets.mvtecloco", "MVTecLocoDataset"],
}


@click.group(chain=True)
@click.option("--results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", is_flag=True)
@click.option("--teacher_epochs", type=int, default=1000)
@click.option("--student_epochs", type=int, default=15)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
    teacher_epochs,
    student_epochs,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"]()

    device = utils.set_torch_device(gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["name"],
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["name"]

        imagesize = dataloaders["training"].dataset.imagesize
        stnet_list = methods["get_stnet"](device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, stnet in enumerate(stnet_list):
            # torch.cuda.empty_cache()
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(stnet_list))
            )
            # torch.cuda.empty_cache()

            stnet.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)
            if not test:
                i_auroc = stnet.train(
                    training_data = dataloaders["training"], 
                    test_data = dataloaders["testing"],
                    teacher_epochs = teacher_epochs,
                    student_epochs = student_epochs,
                    )
            else:
                # BUG: the following line is not using. Set test with True by default.
                i_auroc =  stnet.test(dataloaders["training"], dataloaders["testing"], save_segmentation_images = False)
                print("Warning: Pls set test with true by default")

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": i_auroc, # auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

        LOGGER.info("\n\n-----\n")
        

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

@main.command("stnet")
@click.option("--patch_size", type=int, default=65)
@click.option("--n_students", type=int, default=3)
def stnet(patch_size, 
          n_students):
    def get_stnet(device):
        stnets = []
        stnetInst = src.StudentTeacher.StudentTeacherAnomalyDetector(device=device)
        stnetInst.load(
            patch_size=patch_size,
            n_students=n_students,
        )
        stnets.append(stnetInst)
        return stnets

    return ("get_stnet", get_stnet)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=2, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--patchsize", type=int, default=33)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    imagesize,
    num_workers,
    patchsize=33,
):
    
    def get_dataloaders():
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = AnomalyDataset(
                    root_dir=os.path.join(data_path, subdataset),
                    transform=transforms.Compose([
                        transforms.Resize((imagesize, imagesize)),
                        transforms.RandomCrop((patchsize, patchsize)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(180),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # type: ignore[arg-type]
                    type='train')
            

            test_dataset = AnomalyDataset(
                root_dir=os.path.join(data_path, subdataset),
                transform=transforms.Compose([
                    transforms.Resize((imagesize, imagesize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),# type: ignore[arg-type]
                    gt_transform=transforms.Compose([
                    transforms.Resize((imagesize, imagesize)),
                    transforms.ToTensor()]),# type: ignore[arg-type]
                    type='test') 
                  
            LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

            train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size,
                                shuffle=True, 
                                num_workers=num_workers)
            
            test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=num_workers)
            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset
            
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
