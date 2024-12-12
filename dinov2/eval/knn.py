# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import re
import sys
from functools import partial
from typing import List, Optional
import matplotlib.pyplot as plt
from utils import PCA, IncrementalPCAWrapper, visualize_embeddings, save_embeddings
import wandb
import numpy as np
import torch
from torch.nn.functional import one_hot, softmax
from PIL import Image, ImageOps
import io
from tensorboardX import SummaryWriter
from tensorboard.plugins import projector
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import dinov2.distributed as distributed
from dinov2.data import (
    SamplerType,
    make_data_loader,
    make_dataset,
)
from dinov2.data.transforms import (
    make_classification_eval_transform,
)
from dinov2.eval.metrics import (
    AccuracyAveraging,
    build_topk_accuracy_metric,
)
from dinov2.eval.setup import (
    get_args_parser as get_setup_args_parser,
)
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import (
    ModelWithNormalize,
    evaluate,
    extract_features,
)

logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=19,
        help="Number of workers in DataLoader.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name for the wandb log",
        default="knn_run",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Set number of nodes used.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default=None,
        help="Directory to save TensorBoard embedding projector logs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Flag to save raw images for TensorBoard Embedding Projector",
    )   
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
    )
    return parser

class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(
        self,
        train_features,
        train_labels,
        nb_knn,
        T,
        device,
        num_classes=1000,
    ):
        super().__init__()

        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(
                *broadcast_shape,
                dtype=features_rank.dtype,
                device=self.device,
            )
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(
            neighbors_labels,
            retrieved_rank,
            dst=target_rank,
        )

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(
                neighbors_labels,
                num_classes=self.num_classes,
            ),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(
    *,
    module,
    n_per_class_list,
    n_tries,
    nb_knn,
    train_features,
    train_labels,
):
    print(f"Shape of train_labels: {train_labels.shape}")
    print(f"Shape of train_features: {train_features.shape}")
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    #print("mapping", mapping)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            #print("npc", npc)
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}

def plotting(features, labels, step=0):
    features = features.cpu()
    ipca = IncrementalPCAWrapper(num_components=2, batch_size=1024)  # Adjust batch size if needed
    ipca.fit(features)
    reduced_features = ipca.transform(features)
        

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0].cpu().numpy(),
        reduced_features[:, 1].cpu().numpy(),
        c=labels.cpu().numpy(),
        cmap="tab10",  
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Class Labels")
    plt.title("2D Visualization of Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plot_path = f"plots/step_{step}_features.png"
    plt.savefig(plot_path)
    plt.close()
    
    wandb.log({"Feature Visualization": wandb.Image(plot_path)})

def eval_knn(
    model,
    train_dataset,
    val_dataset,
    accuracy_averaging,
    nb_knn,
    temperature,
    batch_size,
    num_workers,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
    tensorboard_log_dir=None,
    save_images=False,
):
    model = ModelWithNormalize(model)

    logger.info("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model,
        train_dataset,
        batch_size,
        num_workers,
        gather_on_cpu=gather_on_cpu,
    )


    if tensorboard_log_dir is not None:
        wandb_run_name = wandb.run.name if wandb.run else "default_run"
        wandb_log_dir = os.path.join(tensorboard_log_dir, wandb_run_name)
        embeddings_dir = os.path.join(wandb_log_dir, 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)

        metadata_path = os.path.join(embeddings_dir, 'metadata.tsv')
        unique_labels = np.unique(train_labels.cpu().numpy())
        with open(metadata_path, 'w') as f:
            for label in unique_labels:
                f.write(f"{label}\n")

        embedding_tensor = torch.tensor(train_features.cpu().numpy())
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddings'
        embedding.metadata_path = 'metadata.tsv'

        torch.save({'embeddings': embedding_tensor}, embeddings_dir + '/embeddings.pt')

        
        max_sprite_images = 256

        sprite_images = []
        sprite_labels = []
        sprite_indices = []

        class_counts = {}
        for label in train_labels:
            class_label = label.item()
            class_counts[class_label] = class_counts.get(class_label, 0) + 1

        # Calculate sampling probabilities to maintain original distribution
        total_images = len(train_labels)
        sampling_ratios = {}
        for label, count in class_counts.items():
            class_proportion = count / total_images
            class_sprite_images = max(1, int(max_sprite_images * class_proportion))
            sampling_ratios[label] = min(class_sprite_images, count)

        for label, count in sampling_ratios.items():
            label_indices = torch.where(train_labels == label)[0]
            
            if len(label_indices) > count:
                sampled_indices = torch.randperm(len(label_indices))[:count]
                label_indices = label_indices[sampled_indices]
            
            for idx in label_indices:
                image_data = train_dataset.get_image_data(idx)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                target_size = (224, 224)
                resized_image = image.resize(target_size, Image.LANCZOS)
                
                assert resized_image.size == target_size, "Image resizing failed!"
                
                tensor_image = ToTensor()(resized_image)
                sprite_images.append(tensor_image)
                sprite_labels.append(label)
                sprite_indices.append(idx)

        sprite_images = torch.stack(sprite_images)
        sprite_labels = torch.tensor(sprite_labels)

        images_dir = os.path.join(embeddings_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Save sprite images
        image_paths = []
        for i, img in enumerate(sprite_images):
            pil_image = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))  # (H, W, C)
            image_path = os.path.join(images_dir, f"image_{i}.png")
            pil_image.save(image_path)
            image_paths.append(image_path)

        # Prepare embedding configuration
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddings'
        embedding.metadata_path = 'metadata.tsv'
        embedding.sprite.image_path = os.path.relpath(images_dir, embeddings_dir)
        embedding.sprite.single_image_dim.extend([224, 224])


        writer = SummaryWriter(log_dir=embeddings_dir)
        selected_embedding_tensor = embedding_tensor[sprite_indices]

        # Log embeddings to TensorBoard
        writer = SummaryWriter(log_dir=embeddings_dir)
        writer.add_embedding(
            mat=selected_embedding_tensor,  # Use only embeddings for sprite images
            label_img=sprite_images,
            metadata=train_labels[sprite_indices].cpu().tolist(),  # Corresponding labels
            global_step=0
        )
        writer.close()

        print("embedding_tensor shape:", embedding_tensor.shape)
        print("sprite_images shape:", sprite_images.shape)
        print("sprite_indices:", len(sprite_indices))



        # Optional: Save metadata for more detailed inspection
        metadata_path = os.path.join(embeddings_dir, 'metadata.tsv')
        with open(metadata_path, 'w') as f:
            f.write("index\tlabel\tis_sprite\n")
            
            for i in range(len(train_labels)):
                is_sprite = 1 if i in sprite_indices else 0
                f.write(f"{i}\t{train_labels[i].item()}\t{is_sprite}\n")


    logger.info(f"Train features created, shape {train_features.shape}.")
    #plotting(train_features, train_labels) #broken

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )
    num_classes = int(train_labels.max() + 1)
    print("Train num_classes", num_classes)
    metric_collection = build_topk_accuracy_metric(accuracy_averaging, num_classes=num_classes)

    device = torch.cuda.current_device()
    partial_module = partial(
        KnnModule,
        T=temperature,
        device=device,
        num_classes=num_classes,
    )
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            print(f"Output from postprocessor: {postprocessors}")
            metrics = {
                **metrics,
                **{
                    (
                        n_per_class,
                        t,
                        k,
                    ): metric_collection.clone()
                    for k in knn_try.nb_knn
                },
            }
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    _, results_dict = evaluate(
        model_with_knn,
        val_dataloader,
        postprocessors,
        metrics,
        device,
    )

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
                if "confmat" not in key
            }
            if "confmat" in keys:
                results_dict[(n_per_class, k)]["confmat"] = torch.sum(
                    torch.stack([results_dict[(n_per_class, t, k)]["confmat"] for t in knn_module.keys()]),
                    dim=0,
                )

            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


def eval_knn_with_model(
    model,
    output_dir,
    train_dataset_str="ImageNet:split=TRAIN",
    val_dataset_str="ImageNet:split=VAL",
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
    transform=None,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=5,
    n_per_class_list=[-1],
    n_tries=1,
    tensorboard_log_dir=None,
    save_images=False,
):
    transform = transform or make_classification_eval_transform()

    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=transform,
        with_targets=True
    )
    val_dataset = make_dataset(
        dataset_str=val_dataset_str,
        transform=transform,
        with_targets=True
    )

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            accuracy_averaging=accuracy_averaging,
            nb_knn=nb_knn,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
            tensorboard_log_dir=tensorboard_log_dir,
            save_images=save_images,
        )

    results_dict, confmats_dict = {}, {}
    if distributed.is_main_process():
        for knn_ in results_dict_knn.keys():
            metric_log_msg = f"KNN {knn_[1]} classifier result: "
            for metric_name in results_dict_knn[knn_].keys():
                metric_val = results_dict_knn[knn_][metric_name]
                if "confmat" in metric_name:
                    metric_val = metric_val.cpu()
                    confmats_dict[knn_] = np.array(metric_val, dtype=np.uint)
                else:
                    metric_val = metric_val.item()
                    results_dict[f"{knn_} {metric_name}"] = metric_val
                    metric_log_msg += f"{metric_name}: {metric_val:.4f} "
                    # Log metrics to wandb
                    wandb.log({f"{knn_}_{metric_name}": metric_val})
                if "confmat" not in metric_name:
                    logger.info(metric_log_msg)

    # Save evaluation results and confusion matrices
    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    confmat_file_path = os.path.join(output_dir, "confmats_knn")
    os.makedirs(confmat_file_path, exist_ok=True)
    np.save(confmat_file_path + ".npy", confmats_dict)
    for k, v in confmats_dict.items():
        knn_nb = re.search("[0-9]+", str(k))
        if knn_nb:
            knn_nb = knn_nb.group(0)
        else:
            knn_nb = k
        np.save(
            os.path.join(confmat_file_path, f"knn_{knn_nb}"),
            v,
        )
        # Log confusion matrices to wandb
        wandb.log({f"Confusion Matrix KNN {knn_nb}": wandb.Table(dataframe=v)})

    if distributed.is_enabled():
        torch.distributed.barrier()
    return results_dict


# @record
def main(args):
    model, autocast_dtype = setup_and_build_model(args, do_eval=True)

    print("args.output_dir", args.output_dir)
    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=None,
        gather_on_cpu=args.gather_on_cpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
        tensorboard_log_dir=args.tensorboard_log_dir,
        save_images=args.save_images,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 k-NN evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
