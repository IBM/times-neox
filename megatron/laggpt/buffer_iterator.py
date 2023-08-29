
import torch
from megatron import mpu, print_rank_0
from .augmentation import AugmentationIterator


from megatron.mpu import (
    get_model_parallel_rank, 
    get_pipe_parallel_rank, get_pipe_parallel_world_size, 
    broadcast_data_first_last_layer_group
)


#from .datasets import get_combined_dataset
#from .dataloaders import get_train_valid_test_dataloaders
from megatron.mpu.data import broadcast_data_first_last_layer_group, broadcast_keys_first_last_layer_group


def buffer_broadcast_iterator(iterator, size, src):
    pipe_rank = get_pipe_parallel_rank()

    need_data = (pipe_rank == 0 or pipe_rank == get_pipe_parallel_world_size() - 1)
    if not need_data:
        return None
    return BufferBroadcastIterator(iterator, size, src)


class BufferBroadcastIterator:
    """
    Deepspeed pipeline requires data batches on the first and last stages.
    We broadcast batch from the first pipeline stage at first model rank to all model ranks at first and last stages for each data group.
    To avoid interference with DeepSpeed communications, data is buffered in the beginning of training, validation, testing iterations.
    """
    def __init__(self, iterator, size, src):
        self.size = size
        self.iterator = iterator
        self.index  = 0
        self.src = src

        data, keys = None, None
        
        if self.src:
            data = next(self.iterator)
            keys = list(data.keys())
        self.keys = broadcast_keys_first_last_layer_group(keys)
        
        data = broadcast_data_first_last_layer_group(self.keys, data, datatype=torch.float32, src = self.src)
        self.buffer = self._update_buffer(add_data=True, data = data)


    def _update_buffer(self, add_data = False, data = None):
    
        buffer, size = ([data], self.size - 1) if add_data else ([], self.size)   
        
        for _ in range(size):
            data = None
            if self.src:
                data = next(self.iterator)
                if data is None:
                    raise ValueError(f"Data is None. {self.src} and {self.size} and {self.index}")
                
            
            data = broadcast_data_first_last_layer_group(self.keys, data, datatype=torch.float32, src = self.src)
            buffer.append(data)
        return buffer


    def update_buffer(self):
        if self.index == self.size:
            self.buffer = self._update_buffer()
            self.index = 0


    def __iter__(self):
        return self


    def __next__(self):
        batch = self.buffer[self.index]
        self.index += 1
        if batch is None:
            raise ValueError(f"Batch is None. {self.src} and {self.size} and {self.index}")
        return batch


def get_iterator(data):
    return iter(data) if data else None

def buffer_train_valid_test_data_iterators(neox_args, dataloaders):
    """
    Convert dataloaders to iterators adding broadcasting-buffering (see BufferBroadcastIterator) and augmentation 
    """

    print_rank_0("> building train, validation, and test datasets ...")

    src = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
    train, valid, test = dataloaders(neox_args) if src else (None, None, None)
    train, valid, test = get_iterator(train), get_iterator(valid), get_iterator(test)

    dataset_opt = neox_args.times_args["datasets"]
    if train and ("augmentation" in dataset_opt) and dataset_opt["augmentation"]["enabled"]:
        print_rank_0("> data_augmentation set ...")
        train = AugmentationIterator(dataset_opt["augmentation"], train) 

    train_buffer_size = neox_args.gradient_accumulation_steps * neox_args.train_micro_batch_size_per_gpu
    validation_buffer_size = neox_args.gradient_accumulation_steps * neox_args.train_micro_batch_size_per_gpu

    train = buffer_broadcast_iterator(train, train_buffer_size, src) 
    valid = buffer_broadcast_iterator(valid, validation_buffer_size, src)
    

    do_train = train is not None and neox_args.train_iters > 0
    do_valid = valid is not None and neox_args.eval_iters > 0
    do_test = test is not None and neox_args.eval_iters > 0

    flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    torch.distributed.broadcast(flags, src=0)

    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    print_rank_0("> building train, validation, and test datasets done ...")

    return train, valid, test


def buffer_test_data_iterator(neox_args, dataloader):
    print_rank_0("> building test datasets ...")

    src = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
    test = dataloader(neox_args) if src else None
    test = get_iterator(test)

    buffer_size = 1
    test = buffer_broadcast_iterator(test, buffer_size, src) 
    return test
