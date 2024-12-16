# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import time
from collections import defaultdict, deque
from typing import List

import torch
import wandb
import dinov2.distributed as distributed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None, verbose=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file
        self.verbose = verbose

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if "batch_size" in name:
                meter = int(meter.deque[-1])
                name = "b_s"
            if name.endswith("crops_loss"):
                name = name[: -(len("crops_loss") + 1)]
            elif name.endswith("loss"):
                name = name[: -(len("loss") + 1)]

            if "last_layer_lr" in name:
                name = "ll_lr"

            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        pass

    def log_every(
        self,
        iterable,
        print_freq,
        header=None,
        n_iterations=None,
        start_iteration=0,
    ):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f} (mb)"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                if self.verbose:
                    self.dump_in_output_file(
                        iteration=i,
                        iter_time=iter_time.avg,
                        data_time=data_time.avg,
                    )
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.verbose:
            if n_iterations == 0:
                time_per_it = 0
            else:
                time_per_it = total_time / n_iterations
            logger.info("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, time_per_it))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device="cuda",
        )
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def log_images_to_wandb(missclassified_images: List[dict], class_map=None):
    """
    Log images to WandB with predicted and original labels.
    
    Args:
        missclassified_images (list): List of dictionaries containing image, predicted label, and true label.
        class_map (dict): Dictionary mapping class indices to class names.
    """
    images_to_log = []
    for image in missclassified_images:
        print(f"Image size: {image['image'].shape}")
        img = image["image"].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
        pred_label = class_map[image["predicted_label"]] if class_map is not None else image["predicted_label"]
        true_label = class_map[image["true_label"]] if class_map is not None else image["true_label"]

        # Create WandB image with caption
        images_to_log.append(
            wandb.Image(img, caption=f"Pred: {pred_label}, True: {true_label}")
        )

    # Log to WandB
    wandb.log({"predictions": images_to_log})

def get_missclassified_images_for_logging(data, labels, outputs, num_images_to_log=5):
    misclassified_images = []
    preds = torch.argmax(outputs, dim=1)
    # Find misclassified images and store them
    misclassified_indices = (preds != labels).nonzero(as_tuple=True)[0]
    for idx in misclassified_indices[:num_images_to_log]:
        misclassified_images.append({
            "image": data[idx].cpu(),
            "true_label": labels[idx].item(),
            "predicted_label": preds[idx].item(),
        })
    return misclassified_images

def log_confusion_matrix_to_wandb(labels, outputs, class_labels):
    # Compute confusion matrix
    cm = confusion_matrix(labels, outputs, labels=np.arange(len(class_labels)))

    # Create a heatmap with seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, norm=LogNorm(), cbar_kws={'shrink': 0.75} )
    plt.xticks(rotation=90, fontsize=6)  # Smaller, rotated x-axis labels
    plt.yticks(rotation=0, fontsize=6)   # Smaller y-axis labels
    plt.xlabel("Predicted", fontsize=8)
    plt.ylabel("Actual", fontsize=8)
    plt.title("Confusion Matrix", fontsize=10)

    # Save the plot as an image and log it to WandB
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    plt.close()