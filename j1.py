import ray
import jax


@ray.remote(resources={"TPU": 4})
def my_function() -> int:
  return jax.device_count()


num_tpus = ray.available_resources()["TPU"]
num_hosts = int(num_tpus) // 4
h = [my_function.remote() for _ in range(num_hosts)]
print(ray.get(h)) # [16, 16, 16, 16]
