import datasets
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker

class CustomTrainer(Trainer):
    def __init__(self, *args, override_dataloader=False, override_loss=False, num_classes=None, loss_fn=None, **kwargs):
        """
        Custom Trainer class that optionally overrides get_train_dataloader and compute_loss.
        :param override_dataloader: Whether to override the get_train_dataloader method.
        :param override_loss: Whether to override the compute_loss method.
        :param num_classes: Number of classes, used in compute_loss.
        :param loss_fn: Custom loss function.
        """
        super().__init__(*args, **kwargs)
        self.override_dataloader = override_dataloader
        self.override_loss = override_loss
        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def get_train_dataloader(self):
        if self.override_dataloader:
            # Custom implementation of get_train_dataloader
            print("Custom get_train_dataloader is used.")
            
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
        
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                'batch_size': self._train_batch_size,
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params['sampler'] = self._get_train_sampler()
                dataloader_params['drop_last'] = self.args.dataloader_drop_last
                dataloader_params['worker_init_fn'] = seed_worker
                dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor
            
            # Custom sampler.
            def make_weights_for_balanced_classes(labels, num_classes):
                # Count how many times each class occurs across all examples.
                class_counts = torch.zeros(num_classes)
                
                for label_list in labels:
                    # label_list is a list of float, either 0. or 1..
                    for k in range(len(label_list)):
                        class_counts[k] += label_list[k]  # Increment for each class in the label list
                
                # Compute the weight for each class as the inverse of its frequency.
                weight_per_class = class_counts.sum() / class_counts.float()
                
                # Calculate the average weight for each example based on its labels.
                weights = []
                for label_list in labels:
                    label_mask = torch.tensor(label_list)
                    weights.append(torch.sum(weight_per_class * label_mask))  # Average weight for multi-label example
                
                return torch.tensor(weights)
            
            labels = self.train_dataset['labels']
            weights = make_weights_for_balanced_classes(labels, self.num_classes)

            sampler = WeightedRandomSampler(weights, len(weights))
            dataloader_params['sampler'] = sampler

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        else:
            return super().get_train_dataloader()  # Default dataloader method

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.override_loss:
            labels = inputs.get('labels')

            # Forward pass
            outputs = model(**inputs)
            try:
                logits = outputs.get('logits')
            except AttributeError:
                logits = outputs
            
            # Compute loss using the provided loss function (e.g., Focal Loss)
            loss = self.loss_fn(logits, labels)
            
            # Use the provided custom loss function if available
            if self.loss_fn:
                loss = self.loss_fn(logits, labels)
            else:
                raise ValueError("A custom loss function must be provided when override_loss is True.")
            
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs)  # Default loss method