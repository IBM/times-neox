from megatron.laggpt.inference import inference, initialize
from torch.distributed import barrier

neox_args, model, times_envelope, data_iterator = initialize()
inference(neox_args, model, times_envelope, data_iterator)

barrier()