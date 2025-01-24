import torch
import torch_xla
import torch_xla.core.xla_model as xm


def _mp_fn(index):
    t = torch.randn(2, 2, device=xm.xla_device())
    print(t.device)
    print(t)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
