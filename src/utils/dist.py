import os
import torch
import torch.distributed as dist

def setup_dist():
    """
    Initializes the distributed process group.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
        print(f"Distributed Init: Rank {rank}/{world_size}, Local Rank {gpu}, Device: {gpu}")
        return True
    return False

def cleanup_dist():
    """
    Destroys the process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """
    Returns the global rank of the current process.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """
    Returns the total number of processes.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    """
    Returns True if the current process is the main process (rank 0).
    """
    return get_rank() == 0

def reduce_tensor(tensor):
    """
    Reduces the tensor across all processes (averaging).
    
    Args:
        tensor (torch.Tensor): The tensor to reduce.
        
    Returns:
        torch.Tensor: The reduced tensor.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt
