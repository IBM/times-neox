from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.trainingTIMES import setup_model_and_optimizer
from megatron.laggpt.dataloaders import combined_test_dataset_iterator
from megatron import print_rank_0
import megatron.mpu as mpu
import torch
import os, zarr, shutil




class GenerationIterator:
    """
    During inference time, the same data iterators are used as in training and validation stages.
    Initially, the future_target is replaced with a single non-observed point (non-observed values does not affect loss).
    On each iteration, new values are inserted before non-observed point.
    """
    def __init__(self, iterator, times_envelope):
        self.iterator = iterator
        self.times_envelope = times_envelope
        self.src = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
        self.keys = None
        self.ground_truth, self.past_target, self.future = None, None, None

    @torch.no_grad()
    def set_new_vals(self, new_vals):
        self.future = torch.hstack([self.future, new_vals]) if self.future is not None else new_vals

    def get_batch_size(self):
        return self.batch["past_target"].size(0)

    def get_batch_device(self):
        return self.batch["past_target"].device

    def start_new_seq(self):
        self.new_seq = True
        
        self.iterator.update_buffer()
        self.batch = next(self.iterator)
        self.ground_truth = self.batch["future_target"]
        self.past_target = self.batch["past_target"]

        self.loc, self.scale = self.times_envelope.get_loc_scale(self.batch)
        self.future = None

    @torch.no_grad()
    def __next__(self):

        batch = self.batch
      
        past_target = batch["past_target"]
        batch_dim = past_target.size(0)
        future_values = torch.zeros((batch_dim, 1), dtype = past_target.dtype, device = past_target.device)
        future_observed_values = torch.zeros((batch_dim, 1), dtype = past_target.dtype, device = past_target.device)

        if self.future is not None:
            future_values =  torch.hstack([self.future, future_values])
            ones = torch.ones((batch_dim, self.future.size(-1)), dtype = torch.float32, device = past_target.device)
            future_observed_values = torch.hstack([ones, future_observed_values])
        batch["future_target"] = future_values
        batch["future_observed_values"] = future_observed_values

        return batch
    

def initialize():
    """
    An attemnt to rewrite GPT-NeoX options to tailor them for inference stage. 
    TODO: gradient_accumulation_steps override does not work and need to be changed manually in config file.  
    """
    _overwrite_values = {
        "gradient_accumulation_steps": 1,
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "optimizer": None,  # prevent loading optimizer (no_load_optim alone won't work)
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=_overwrite_values)
    neox_args.configure_distributed_args()
    
    

    # initialize megatron
    initialize_megatron(neox_args)
    model, times_envelope , _, _ = setup_model_and_optimizer(neox_args, use_cache = False)
    data_iterator = combined_test_dataset_iterator(neox_args)
    return neox_args, model, times_envelope, data_iterator


def process_batch(model, times_envelope, gen_iterator, prediction_length):
    
    data_nodes = (model.is_first_stage() or model.is_last_stage())
    pipe_rank_shift = (mpu.get_pipe_parallel_world_size() - 1) * mpu.get_model_parallel_world_size() * mpu.get_data_parallel_world_size()

    if data_nodes:
        gen_iterator.start_new_seq()
    
    #model.module.clear_cache()
    for i in range(prediction_length):

        # clear tensor metadata
        model.first_output_send = True
        model.pipe_recv_buf = None

        print_rank_0(">    Token: ", i)
        _, outputs = model.eval_batch(gen_iterator, return_logits = True)
        
        if model.is_last_stage():
            last_column = [i[:, -1:, ...] for i in outputs]
            loc, scale = gen_iterator.loc, gen_iterator.scale
            new_vals = times_envelope.get_greedy_val(last_column, loc, scale)
            torch.distributed.send(new_vals, torch.distributed.get_rank() - pipe_rank_shift)
        if model.is_first_stage():
            new_vals = torch.empty((gen_iterator.get_batch_size(), 1), dtype = torch.float32, device = gen_iterator.get_batch_device())
            torch.distributed.recv(new_vals, torch.distributed.get_rank() + pipe_rank_shift)

        if data_nodes:
            gen_iterator.set_new_vals(new_vals)

    ground_truth = to_numpy(gen_iterator.ground_truth)
    past_target = to_numpy(gen_iterator.past_target)
    future = to_numpy(gen_iterator.future)

    return ground_truth, past_target, future


def inference(neox_args, model, times_envelope, data_iterator):
    """
    Main inference routine. 
    TODO: need to be changed to handle larger output, caching is not tested and turned off for now.
    """

    times_args = neox_args.times_args
    inference_opt = times_args["inference"]
    chunk_size = inference_opt["chunk_size"]
    prediction_length = times_args["prediction_length"]
    gen_iterator = GenerationIterator(data_iterator, times_envelope)
    output_filename = inference_opt["file_name"]
    
    if torch.distributed.get_rank() == 0:
        if os.path.exists(output_filename) and os.path.isdir(output_filename):
            shutil.rmtree(output_filename)

    torch.distributed.barrier()

    write_nodes = (mpu.get_pipe_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)

    if write_nodes:
        root = zarr.open(output_filename)
        group = root.create_group(f"data{mpu.get_data_parallel_rank():04d}.json")

    def create_array(name, array):
        return group.array(name, array, chunks = (chunk_size, array.shape[-1]))

    for i in range(inference_opt["num_test_batches"]):
        print_rank_0("> Batch index: ", i)
        ground_truth, past_target, future = process_batch(model, times_envelope, gen_iterator, prediction_length)
        if write_nodes:
            if i == 0:
                ground_truth_array = create_array("ground_truth", ground_truth)
                past_target_array = create_array("past_target", past_target)
                future_array = create_array("future", future)
            else:
                ground_truth_array.append(ground_truth)
                past_target_array.append(past_target)
                future_array.append(future)

    


def to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.cpu().numpy()

