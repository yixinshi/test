import ray
import jax
import pprint 

# To submit the job to Ray:
# ray job submit --runtime-env-json='{"working_dir": "."}' -- python j1.py
# Or run it directly at the head node.

ray.init()

@ray.remote(resources={"TPU": 4})
def my_function() -> int:
  return [str(x) for x in (jax.devices(), jax.device_count(), jax.local_devices(), jax.process_indices(), jax.process_index(), jax.process_count())]


num_tpus = ray.available_resources()["TPU"]
print("number of tpus: %d" % num_tpus)
num_hosts = int(num_tpus) // 4
h = [my_function.remote() for _ in range(num_hosts)]

print("jax.devices(), jax.device_count(), jax.local_devices(), jax.process_indices(), jax.process_index(), jax.process_count()")
pprint.pp(ray.get(h)) # [16, 16, 16, 16]
