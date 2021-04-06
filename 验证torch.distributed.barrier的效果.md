# 验证torch.distributed.barrier的效果

最终可以发现，对于每个不同的进程，barrier的作用就是：
**所有进程必须执行都按照对应次序来执行`dist.barrier`后（位置可以不同，但是必须要有，不然会最终陷入等待），才会继续执行之后的动作**。

```python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import time

import torch.distributed as dist
import torch.multiprocessing as mp


def ddp_test_v0(local_rank, word_size):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend="nccl", world_size=word_size, rank=local_rank)

    print("first before barrier{}\n".format(local_rank))
    if local_rank != 0:
        dist.barrier()
    print("first after barrier{}\n".format(local_rank))

    print("inter {}".format(local_rank))

    print("second before barrier{}\n".format(local_rank))
    if local_rank == 0:
        dist.barrier()
    print("second after barrier{}\n".format(local_rank))

    print("{} exit".format(local_rank))


def ddp_test_v1(local_rank, word_size):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend="nccl", world_size=word_size, rank=local_rank)

    if local_rank != 0:
        print("1 before barrier{}\n".format(local_rank))
        start = time.time()
        time.sleep(5)
        dist.barrier()
        print(time.time() - start)
        print("1 after barrier{}\n".format(local_rank))
        # dist.barrier()
        print("1 after barrier{}\n".format(local_rank))
    else:
        print("0 before barrier{}\n".format(local_rank))
        start = time.time()
        time.sleep(3)
        dist.barrier()
        print(time.time() - start)
        print("0 after barrier{}\n".format(local_rank))
        print("0 after barrier{}\n".format(local_rank))
        dist.barrier()
        print("0 after barrier{}\n".format(local_rank))

    print("{} exit".format(local_rank))


def main():
    world_size = 2
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(ddp_test_v1, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()
```
