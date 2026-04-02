import torch
from torch.utils.data import DataLoader
import yaml
from .dataset import create_ETHUSDT_dataset
dataset_config_path = '/root/lio/modelresearch/Configs/experiment_config.yaml'
model_config_path = '/root/lio/modelresearch/Configs/model_config.yaml'
with open(dataset_config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)



dataloader_config = dataset_config.get('dataloader', {})

dataset_func = create_ETHUSDT_dataset

class ETHUSDTDataLoaders:
    def __init__(self):
        super().__init__()
        self.batch_size = dataloader_config.get('batch_size', 128)
        self.num_workers = dataloader_config.get('num_workers', 4)
        self.pin_memory = dataloader_config.get('pin_memory', True)
        self.drop_last = dataloader_config.get('drop_last', True)
        self.prefetch_factor = dataloader_config.get('prefetch_factor', 4)

        self.train = self._make_dataloader('train')
        self.val = self._make_dataloader('val')

    def _make_dataloader(self,split):
        
        return DataLoader(dataset_func(split=split),
                                batch_size=self.batch_size,
                                shuffle= split=='train',
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                prefetch_factor=self.prefetch_factor,
                                drop_last=self.drop_last)

