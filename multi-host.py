import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import ray

import torch.distributed as dist
import torch_xla.runtime as xr
from torch_xla._internal import pjrt

os.environ["RAY_DEDUP_LOGS"] = "0"

WORLD_SIZE = 8


def init_env():
  local_rank = int(os.environ['TPU_VISIBLE_CHIPS'])
  local_world_size = WORLD_SIZE

  pjrt.initialize_multiprocess(local_rank, local_world_size)
  xr._init_world_size_ordinal()


@ray.remote(resources={"TPU": 1})
def print_tensor():
  init_env()

  t = torch.randn(2, 2, device=xm.xla_device())
  print(t.device)
  print(t)


ray.init()

tasks = [print_tensor.remote() for _ in range(WORLD_SIZE)]
ray.get(tasks)

ray.shutdown()
