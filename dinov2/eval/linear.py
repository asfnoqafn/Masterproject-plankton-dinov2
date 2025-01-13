# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import List, Optional
import wandb
import numpy as np
import torch
import torch.nn as nn
from fvcore.common.checkpoint import (
    Checkpointer,
    PeriodicCheckpointer,
)
from torch.nn.parallel import DistributedDataParallel
import dinov2.logging.helpers as logging_helpers
import dinov2.distributed as distributed
from dinov2.data import (
    SamplerType,
    make_data_loader,
    make_dataset,
)
from dinov2.data.transforms import (
    make_classification_eval_transform,
    make_classification_train_transform,
)
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import (
    get_args_parser as get_setup_args_parser,
)
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import (
    ModelWithIntermediateLayers,
    evaluate,
    load_hierarchy_from_file,
)
from dinov2.logging import MetricLogger

logger = logging.getLogger("dinov2")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        "--run_name",
        type=str,
        help="Name for the wandb log",
        default="lin_run",
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
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--optimizer-momentum",
        type=float,
        default=0.9,
        help="Momentum for the linear classifier optimizer",
    )
    parser.add_argument(
        "--n-last-blocks",
        type=int,
        default=4,
        help="Number of blocks to use for the linear classifier",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the linear classifier",
    )
    parser.add_argument(
        "--avg-pool",
        type=bool,
        default=True,
        help="Whether to use average pooling for the linear classifier",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Set number of nodes used.",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for the linear classifier.",
    )
    parser.add_argument(
        "--use-nesterov",
        type=bool,
        default=True,
        help="Whether to use Nesterov momentum",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--log-missclassified-images",
        type=bool,
        help="This flag enables logging of misclassified images to WandB",
    )
    parser.add_argument(
        "--log-confusion-matrix",
        type=bool,
        help="This flag enables logging of the confusion matrix to WandB",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="cross_entropy",
        help="Loss function to use for training the linear classifier, can be 'cross_entropy' or 'custom_hierarchical'",
    )
    parser.add_argument(
        "--hierarchy-file-path",
        type=str,
        help="Path to the hierarchy file for the custom hierarchical loss function",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        epochs=10,
        batch_size=128,
        num_workers=8,
        epoch_length=1250,
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        learning_rate=5e-3,
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (
            image,
            np.pad(
                targets,
                (0, maxlen - len(targets)),
                constant_values=-1,
            ),
        )
        for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat(
        [class_token for _, class_token in intermediate_output],
        dim=-1,
    )
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
        self,
        out_dim,
        use_n_blocks,
        use_avgpool,
        num_classes=1000,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(
            x_tokens_list,
            self.use_n_blocks,
            self.use_avgpool,
        )
        return self.linear(output)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer(
            "class_mapping",
            (None if class_mapping is None else torch.LongTensor(class_mapping)),
        )

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": (preds[:, self.class_mapping] if self.class_mapping is not None else preds),
            "target": targets,
        }


def scale_lr(learning_rate, batch_size):
    return learning_rate * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifier(
    sample_output,
    n_last_blocks,
    avg_pool,
    num_classes=1000,
):
    out_dim = create_linear_input(
        sample_output,
        use_n_blocks=n_last_blocks,
        use_avgpool=avg_pool,
    ).shape[1]
    linear_classifier = LinearClassifier(
        out_dim,
        use_n_blocks=n_last_blocks,
        use_avgpool=avg_pool,
        num_classes=num_classes,
    )
    linear_classifier = linear_classifier.cuda()
    if distributed.is_enabled():
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier)
    return linear_classifier

def setup_optimizer(
    model,
    learning_rate,
    weight_decay,
    use_nesterov=True,
    batch_size=256,
    optimizer_momentum=0.9,
):
    scaled_learning_rate = scale_lr(learning_rate, batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_learning_rate,
        momentum=optimizer_momentum,
        weight_decay=weight_decay,
        nesterov=use_nesterov,
    )
    return optimizer

@torch.no_grad()
def evaluate_linear_classifier(
    feature_model,
    linear_classifier,
    data_loader,
    metric_type,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    class_mapping=None,
):
    logger.info("Running validation...")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    num_class_mapping = np.array(class_mapping[:, 1], dtype=int) if class_mapping is not None else None
    metric = build_metric(metric_type, num_classes=num_classes)
    postprocessor = LinearPostprocessor(linear_classifier, num_class_mapping)
    metrics = metric.clone()

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessor,
        metrics,
        torch.cuda.current_device(),
    )

    # Log metrics for the single classifier
    metrics_to_log = {k: np.round(v.float().cpu(), 4) for k, v in results_dict_temp.items()}
    logger.info(f"{prefixstring} -- {metrics_to_log}")
    
    # Log metrics to WandB
    wandb.log({f"{prefixstring}/{k}": v for k, v in metrics_to_log.items()})

    
    # Save metrics to file
    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for classifier_string, metrics in results_dict_temp.items():
                # Convert tensors to lists if needed
                if isinstance(metrics, torch.Tensor):
                    metrics = metrics.cpu().numpy().tolist()  # Convert to a JSON-compatible format
                f.write(json.dumps({classifier_string: metrics}) + "\n")

    return results_dict_temp



def eval_linear(
    *,
    feature_model,
    linear_classifier,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,
    running_checkpoint_period,
    eval_period,
    metric_type,
    training_num_classes,
    resume=True,
    classifier_fpath=None,
    val_class_mapping=None,
    num_images_to_log=20,  # Number of misclassified images to log
    log_missclassified_images=False,
    log_confusion_matrix=False,
    loss_function="cross_entropy",
    hierarchy_file_path=None,
):
    checkpointer = Checkpointer(
        linear_classifier,
        output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)

    logger.info(f"Starting training from iteration {start_iter}")
    metric_logger = MetricLogger(delimiter="  ", verbose=distributed.is_main_process())
    iteration = start_iter

    for data, labels in metric_logger.log_every(train_data_loader, 20, "Training", max_iter, start_iter):
        data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        # Forward pass
        features = feature_model(data)
        outputs = linear_classifier(features)
        if(loss_function == "cross_entropy"):
            loss = nn.CrossEntropyLoss()(outputs, labels)
        if(loss_function == "custom_hierarchical"):
            try:
                hierarchy_root = load_hierarchy_from_file(hierarchy_file_path)
            except: 
                raise ValueError(f"Hierarchy file {hierarchy_file_path} could not be loaded. Please make sure the file exists and is in the correct format.")
            loss = hierarchical_loss(outputs, labels, hierarchy_root, val_class_mapping)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log metrics
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            loss_value = loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss_value, lr=current_lr)

            # WandB logging
            wandb.log({"loss": loss_value, "lr": current_lr, "iteration": iteration})

        # Periodic checkpointing
        if iteration > start_iter + 5 and iteration % running_checkpoint_period == 0:
            if distributed.is_main_process():
                logger.info("Checkpointing running checkpoint")
                periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)

        periodic_checkpointer.step(iteration)

        # Periodic evaluation
        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            evaluate_linear_classifier(
                feature_model=feature_model,
                linear_classifier=remove_ddp_wrapper(linear_classifier),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,
            )

        iteration += 1

    # Final evaluation after training
    val_results_dict = evaluate_linear_classifier(
        feature_model=feature_model,
        linear_classifier=remove_ddp_wrapper(linear_classifier),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
        class_mapping=val_class_mapping,
    )

    # added part for logging misclassified images and confusion matrix
    misclassified_images = []
    all_preds = []
    all_labels = []

    for data, labels in val_data_loader:
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Collect predictions and true labels
        all_labels.extend(labels.cpu().numpy())
        features = feature_model(data)
        outputs = linear_classifier(features)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

        # Get misclassified images
        misclassified_images.extend(
            logging_helpers.get_missclassified_images_for_logging(data, labels, outputs, num_images_to_log)
        )

    # Log misclassified images to WandB
    if(log_missclassified_images):
        logging_helpers.log_images_to_wandb(misclassified_images, class_map=val_class_mapping)

    if(log_confusion_matrix):
        logging_helpers.log_confusion_matrix_to_wandb(all_labels, all_preds, val_class_mapping)

    return val_results_dict, feature_model, linear_classifier, iteration

def hierarchical_loss(predictions, labels, hierarchy_root, val_class_mapping):
    """
    Compute hierarchical loss with tree structure.
    
    :param predictions: Tensor of shape (N, C) with predicted probabilities for each class.
    :param labels: Tensor of shape (N,) with ground-truth class indices.
    :param hierarchy_root: Root of the hierarchy tree.
    :param class_indices: Dict mapping class names to indices.
    :return: Computed loss value.
    """
    # Create class mapping dictionaries
    class_indices = {row[0]: int(row[1]) for row in val_class_mapping}

    index_to_class = {v: k for k, v in class_indices.items()}
    
    loss = 0.0
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        true_class = index_to_class[label.item()]
        pred_probs = torch.softmax(pred, dim=0)
        
        # Get the predicted class and its probability
        predicted_class = index_to_class[torch.argmax(pred).item()]
        predicted_prob = pred_probs[torch.argmax(pred)]
        
        # Traverse the hierarchy to determine penalties
        def find_node(node, target):
            if node.name == target:
                return node
            for child in node.children:
                result = find_node(child, target)
                if result:
                    return result
            return None

        true_node = find_node(hierarchy_root, true_class)
        pred_node = find_node(hierarchy_root, predicted_class)
        
        if pred_node is None or true_node is None:
            raise ValueError(f"Class {true_class} or {predicted_class} not found in hierarchy.")
        
        if pred_node == true_node:
            # Correct prediction
            penalty = 0.0
        elif true_node.is_descendant(predicted_class):
            # Predicted class is a parent of the true class
            penalty = 0.1
        elif pred_node.is_descendant(true_class):
            # Predicted class is a child of the true class
            penalty = 0.2
        else:
            # Predicted class is a sibling or unrelated
            penalty = 1.0

        # Combine penalty with negative log likelihood
        loss += penalty * -torch.log(predicted_prob)
    
    return loss / len(predictions)

def make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type):
    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=make_classification_eval_transform(),
        with_targets=True,
    )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=(_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None),
    )
    return test_data_loader


def test_on_datasets(
    feature_model,
    linear_classifier,
    test_dataset_strs,
    batch_size,
    num_workers,
    test_metric_types,
    metrics_file_path,
    training_num_classes,
    iteration,
    best_classifier_on_val,
    prefixstring="",
    test_class_mappings=[None],
):
    results_dict = {}
    for test_dataset_str, class_mapping, metric_type in zip(
        test_dataset_strs,
        test_class_mappings,
        test_metric_types,
    ):
        logger.info(f"Testing on {test_dataset_str}")
        test_data_loader = make_eval_data_loader(
            test_dataset_str,
            batch_size,
            num_workers,
            metric_type,
        )
        dataset_results_dict = evaluate_linear_classifier(
            feature_model,
            remove_ddp_wrapper(linear_classifier),
            test_data_loader,
            metric_type,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring="",
            class_mapping=class_mapping,
        )
        results_dict[f"{test_dataset_str}_accuracy"] = 100.0 * dataset_results_dict["best_classifier"]["accuracy"]
    return results_dict


def run_eval_linear(
    model,
    output_dir,
    train_dataset_str,
    val_dataset_str,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rate,
    autocast_dtype,
    optimizer_momentum,
    weight_decay,
    n_last_blocks,
    avg_pool,
    use_nesterov,
    test_dataset_strs=None,
    resume=True,
    classifier_fpath=None,
    val_class_mapping_fpath=None,
    test_class_mapping_fpaths=[None],
    val_metric_type=MetricType.MEAN_ACCURACY,
    test_metric_types=None,
    log_missclassified_images=False,
    log_confusion_matrix=False,
    loss_function="cross_entropy",
    hierarchy_file_path=None,
):
    seed = 0
    test_dataset_strs = test_dataset_strs or [val_dataset_str]
    test_metric_types = test_metric_types or [val_metric_type] * len(test_dataset_strs)
    assert len(test_dataset_strs) == len(test_class_mapping_fpaths)

    train_transform = make_classification_train_transform()
    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=train_transform,
        with_targets=True,
    )
    # sampler_type = SamplerType.SHARDED_INFINITE
    # sampler_type = SamplerType.INFINITE

    # Determine number of classes
    targets = torch.tensor(train_dataset.get_targets(), dtype=torch.int64)
    training_num_classes = torch.unique(targets).max().item() + 1

    # Setup feature extraction model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    # Setup linear classifier and optimizer
    linear_classifier = setup_linear_classifier(
        sample_output,
        n_last_blocks,
        avg_pool,
        training_num_classes,
    )

    # Setup optimizer
    optimizer = setup_optimizer(
        model=linear_classifier,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_nesterov=use_nesterov,
        batch_size=batch_size,
        optimizer_momentum=optimizer_momentum,
    )

    # Scheduler and checkpoint setup
    max_iter = epochs * epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifier, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    # Data loaders
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=SamplerType.SHARDED_INFINITE,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    val_data_loader = make_eval_data_loader(val_dataset_str, batch_size, num_workers, val_metric_type)

    if(val_class_mapping_fpath.endswith(".json")):
        # Class mapping
        with open(val_class_mapping_fpath, 'r') as f:
            data = json.load(f)

        # If you need it as a NumPy array (for example, for integer mapping):
        val_class_mapping= np.array(list(data.items()))
    else: 
        val_class_mapping = np.load(val_class_mapping_fpath) if val_class_mapping_fpath else None
    test_class_mappings = [
        np.load(fpath) if fpath and fpath != "None" else None for fpath in test_class_mapping_fpaths
    ]
    print("Val class mapping: ", val_class_mapping)

    # Evaluation
    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict = eval_linear(
        feature_model=feature_model,
        linear_classifier=linear_classifier,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=save_checkpoint_frequency * epoch_length,
        running_checkpoint_period=epoch_length,
        eval_period=eval_period_iterations,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        resume=resume,
        val_class_mapping=val_class_mapping,
        num_images_to_log=50,
        log_missclassified_images=log_missclassified_images,
        log_confusion_matrix=log_confusion_matrix,
        loss_function=loss_function,
        hierarchy_file_path=hierarchy_file_path,
    )

    # Test on additional datasets
    if test_dataset_strs != [val_dataset_str]:
        test_on_datasets(
            feature_model,
            linear_classifier,
            test_dataset_strs,
            batch_size,
            num_workers,
            test_metric_types,
            metrics_file_path,
            training_num_classes,
            val_results_dict["iteration"],
            prefixstring="",
            test_class_mappings=test_class_mappings,
        )

    logger.info(f"Validation Results: {val_results_dict}")
    return val_results_dict



def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    print(f"Output dir: {args.output_dir}")
    run_eval_linear(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_strs=args.test_dataset_strs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        n_last_blocks=args.n_last_blocks,
        learning_rate=args.learning_rate,
        optimizer_momentum=args.optimizer_momentum,
        avg_pool=args.avg_pool,
        weight_decay=args.weight_decay,
        use_nesterov=args.use_nesterov,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
        test_metric_types=args.test_metric_types,
        val_class_mapping_fpath=args.val_class_mapping_fpath,
        test_class_mapping_fpaths=args.test_class_mapping_fpaths,
        log_missclassified_images=args.log_missclassified_images,
        log_confusion_matrix=args.log_confusion_matrix,
        loss_function=args.loss_function,
        hierarchy_file_path=args.hierarchy_file_path,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
