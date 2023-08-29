from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

n_pipe = 10
n_model = 5
n_data = 2
size = n_pipe * n_model * n_data

topo = PipeModelDataParallelTopology(num_pp=n_pipe, num_mp=n_model, num_dp=n_data)


def get_model_parallel_src_rank(rank, model_size):
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""

    global_rank = rank
    local_world_size = model_size
    return (global_rank // local_world_size) * local_world_size


groups = []
srcs  = []
n_pipe = topo.get_dim("pipe")
for i in range(topo.get_dim("data")):
    first = topo.filter_match(pipe = 0, data = i)
    last = topo.filter_match(pipe = n_pipe - 1, data = i)
    src = topo.get_rank(pipe = 0, model = 0, data = i)
    group = first + last
    groups.append(group)
    srcs.append(src)

print(groups)
print(srcs)
print(get_model_parallel_src_rank(rank = 7, model_size=n_model))




def _prepare_size_buffer(keys, data, src):

    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if src:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    return sizes_cuda


def unpack_size(keys, sizes_cuda):

    max_dim = _MAX_DATA_DIM
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def _prepare_data_buffer(keys, data, datatype, total_numel, src):
    if src:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0
        ).cuda()
    else:
        flatten_data = torch.empty(
            total_numel, device=torch.cuda.current_device(), dtype=datatype
        )

    return flatten_data


def _unpack_data_buffer(flatten_data, keys, key_size, key_numel):
    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def _cross_broadcast(buffer, model_parallel_world_size, src):
    if src:
        torch.distributed.isend(
            buffer, get_pipe_parallel_last_rank(), 
            group = get_pipe_parallel_group())
    else:
        torch.distributed.recv(
            buffer, get_pipe_parallel_src_rank(),
            group = get_pipe_parallel_group())
    
    if model_parallel_world_size > 1:
        torch.distributed.broadcast(
            buffer, get_model_parallel_src_rank(), group=get_model_parallel_group()
        )


def broadcast_data_ext(keys, data, datatype):
    
    pipe_parallel_world_size = get_pipe_parallel_world_size()
    model_parallel_world_size = get_model_parallel_world_size()

    if model_parallel_world_size < 2 and pipe_parallel_world_size < 2:
        for i in data.keys():
            data[i] = data[i].cuda()
        return data

    if pipe_parallel_world_size < 2:
        return broadcast_data(keys, data, datatype)

    pipe_parallel_rank = get_pipe_parallel_rank()
    model_parallel_rank = get_model_parallel_rank()



    src = (model_parallel_rank == 0 and pipe_parallel_rank == 0)
    sizes_cuda = _prepare_size_buffer(keys, data, src)

    _cross_broadcast(sizes_cuda, model_parallel_world_size, src)

    key_size, key_numel, total_numel = unpack_size(keys, sizes_cuda)
    flatten_data = _prepare_data_buffer(keys, data, datatype, total_numel, src)

    _cross_broadcast(flatten_data, model_parallel_world_size, src)

    return _unpack_data_buffer(flatten_data, keys, key_size, key_numel)
