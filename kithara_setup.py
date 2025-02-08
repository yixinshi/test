import ray
import subprocess

# Initialize Ray
ray.init(address='auto')  # Automatically connects to the cluster if you're on the head node
resources = ray.cluster_resources()
print(resources)

num_nodes = len(ray.nodes())
print(f"Number of nodes in the cluster: {num_nodes}")

# Example usage
cmd = """rm -rf /tmp/test* &&
         mkdir -p /tmp/test && cd /tmp/test &&
         git clone --recursive https://github.com/wenxindongwork/keras-tuner-alpha.git &&
         cd keras-tuner-alpha &&
         git checkout feb5_snapshot &&
         pip install -r requirements.txt && 
         pip install libtpu-nightly==0.1.dev20241010+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      """ 
#pip install -e ./maxtext --no-deps &&
# Submit the task to run remotely on the cluster (head node or workers)

num_tpus = ray.available_resources()["TPU"]
num_hosts = int(num_tpus) // 4
print("number of tpus: {} and hosts: {}".format(num_tpus, num_hosts))


# Define the remote function to install a Python package
# @ray.remote(resources={"TPU": 4})
@ray.remote(resources={"TPU": 4})
def install_dependencies(cmd):
    try:
        # Run pip install using subprocess
        result = subprocess.run(
            [cmd],
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Package '{cmd}' installed successfully!"
        else:
            return f"Error installing package '{cmd}': {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"

future = [install_dependencies.remote(cmd) for _ in range(num_hosts)]

# Wait for the result
result = ray.get(future)
print(result)

# Shut down Ray when done
ray.shutdown()
