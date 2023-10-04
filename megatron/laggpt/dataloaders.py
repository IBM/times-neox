from gluonts.itertools import Cyclic
from typing import Optional, Iterable, Dict, Any
from functools import partial

import torch
import numpy as np
import megatron.mpu as mpu
import random

from .datasets import get_combined_dataset
from .buffer_iterator import buffer_train_valid_test_data_iterators, buffer_test_data_iterator


from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic


from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import (
    Chain,
    Transformation,
    ValidationSplitSampler,
    TestSplitSampler,
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    DummyValueImputation,
    InstanceSampler,
    InstanceSplitter,
)


PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]

def create_instance_splitter(
        sampler, prediction_length, past_length, padding_value):
        
    return InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=sampler,
        past_length=past_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
        dummy_value=padding_value,
    )


def create_data_loader(
        data: Dataset, 
        batch_size, 
        num_batches_per_epoch,
        field_names,
        shuffle_buffer_length: Optional[int] = None):
        
    batches = as_stacked_batches(
        data,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=field_names,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    return batches


def get_training_data_loader(
        data: Dataset, batch_size, num_batches_per_epoch, shuffle_buffer_length, 
        past_length, prediction_length, padding_value):
    
    transform = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
        imputation_method=DummyValueImputation(0.0))
    
    data = transform.apply(data, is_train=True)
    
    sampler = ExpectedNumInstanceSampler(num_instances=1.0, min_instances=1, min_future=prediction_length)
    instance_splitter = create_instance_splitter(sampler, prediction_length, past_length, padding_value)
    data = instance_splitter.apply(Cyclic(data).stream(), is_train=True)
    
    data_loader = create_data_loader(
        data, batch_size, 
        num_batches_per_epoch, TRAINING_INPUT_NAMES, shuffle_buffer_length)
    return data_loader


def get_validation_data_loader(
        data: Dataset, batch_size, num_batches_per_epoch, shuffle_buffer_length, 
        past_length, prediction_length, padding_value):
    
    transform = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
        imputation_method=DummyValueImputation(0.0))
    
    data = transform.apply(data, is_train=True)
    
    sampler = ValidationSplitSampler(min_future=prediction_length) 
    instance_splitter = create_instance_splitter(sampler, prediction_length, past_length, padding_value)
    data = instance_splitter.apply(Cyclic(data).stream(), is_train=True)
    
    data_loader = create_data_loader(
        data, batch_size, 
        num_batches_per_epoch, TRAINING_INPUT_NAMES, shuffle_buffer_length)
    return data_loader


def get_test_gluonts_dataloader(
        data: Dataset, batch_size, num_batches_per_epoch, shuffle_buffer_length, 
        past_length, prediction_length, padding_value):
    
    transform = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
        imputation_method=DummyValueImputation(0.0))
    
    data = transform.apply(data, is_train=True)
    
    sampler = ValidationSplitSampler(min_future=prediction_length) 
    instance_splitter = create_instance_splitter(sampler, prediction_length, past_length, padding_value)
    data = instance_splitter.apply(Cyclic(data).stream(), is_train=True)
    
    data_loader = create_data_loader(
        data, batch_size, 
        num_batches_per_epoch, TRAINING_INPUT_NAMES, shuffle_buffer_length)
    return data_loader




def get_train_valid_test_dataloaders(neox_args, train_dataset, validation_dataset):
    """
    Creates dataloaders from the datasets. Preprocessing time-series: samples with instance samplers, 
    create new fields and flags. 
    """
    
    times_args = neox_args.times_args

    shuffle_buffer_length = times_args["shuffle_buffer_length"]
    prediction_length = times_args["prediction_length"]
    padding_value = times_args["padding_value"]
    past_length = times_args["past_length"]
    
    batch_size = neox_args.train_micro_batch_size_per_gpu
    num_batches_per_epoch = neox_args.train_iters * neox_args.gradient_accumulation_steps
    

    training_data_loader = get_training_data_loader(
        train_dataset, batch_size, num_batches_per_epoch, shuffle_buffer_length, 
        past_length, prediction_length, padding_value)
    
    validation_data_loader = get_training_data_loader(
        validation_dataset, batch_size, num_batches_per_epoch, shuffle_buffer_length, 
        past_length, prediction_length, padding_value)
    
    return training_data_loader, validation_data_loader, None
    
    
def combined_dataset_iterator(neox_args):
    """
    The central function to create a dataset iterator for training from the list of GluonTS datasets.
    Gets combined dataset from GluonTS create dataloaders and creates iterator from dataloaders.
    """
    src = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
    times_args = neox_args.times_args    
    datasets = times_args["datasets"]
    train_datasets = datasets["train"]
    validation_datasets = datasets["validation"]
    
    preload_datasets(train_datasets + validation_datasets)

    if src:
        rank = mpu.get_data_parallel_rank()
        iteration_index = neox_args.iteration
        data_seed = times_args["data_seed"] + iteration_index
        np.random.seed(data_seed)
        random.seed(data_seed)
        train_datasets = get_combined_dataset(train_datasets, rank, data_seed)    
        validation_datasets = get_combined_dataset(validation_datasets, rank, data_seed)

        dataloaders = partial(get_train_valid_test_dataloaders, train_dataset = train_datasets, validation_dataset = validation_datasets)
    else:
        dataloaders = None

    train, validation, test = buffer_train_valid_test_data_iterators(neox_args, dataloaders)
    #TODO Test is not supported yet
    return train, validation, test


def get_test_dataloader(neox_args, dataset):
    """
    Dataloader for test stage
    """
    times_args = neox_args.times_args

    shuffle_buffer_length = times_args["shuffle_buffer_length"]
    prediction_length = times_args["prediction_length"]
    padding_value = times_args["padding_value"]
    past_length = times_args["past_length"]
    n_batches = times_args["inference"]["num_test_batches"]
    
    batch_size = neox_args.train_micro_batch_size_per_gpu
    
    test_dataloader = get_test_gluonts_dataloader(
        dataset, batch_size, n_batches, shuffle_buffer_length, 
        past_length, prediction_length, padding_value)
    
    return test_dataloader
    
def preload_datasets(datasets):

    if torch.distributed.get_rank() == 0:
        for i in datasets:
            get_dataset(i)

    torch.distributed.barrier()

def combined_test_dataset_iterator(neox_args):
    """
    The central function to create a dataset iterator for testing from the list of GluonTS datasets.
    Gets combined dataset from GluonTS create dataloaders and creates iterator from dataloaders.
    """

    src = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
    times_args = neox_args.times_args
    test_datasets = times_args["datasets"]["test"]
    preload_datasets(test_datasets)

    if src:
        rank = mpu.get_data_parallel_rank()
        dataset = get_combined_dataset(test_datasets, rank, times_args["data_seed"], test = True)
        dataloader = partial(get_test_dataloader, dataset = dataset)
    else:
        dataloader = None

    test = buffer_test_data_iterator(neox_args, dataloader)
    return test








