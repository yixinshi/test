import ray
import subprocess

# Initialize Ray
ray.init(address='auto')  # Automatically connects to the cluster if you're on the head node
resources = ray.cluster_resources()
print(resources)

num_nodes = len(ray.nodes())
print(f"Number of nodes in the cluster: {num_nodes}")

# Define the remote function to install a Python package
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

# Example usage
cmd = """rm -rf /tmp/test* &&
         mkdir -p /tmp/test && cd /tmp/test &&
         git clone --recursive https://github.com/wenxindongwork/keras-tuner-alpha.git &&
         cd keras-tuner-alpha &&
         pip install -r requirements.txt && 
         pip install -e ./maxtext --no-deps &&
         pip install libtpu-nightly==0.1.dev20241010+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      """ 
# Submit the task to run remotely on the cluster (head node or workers)
future = [install_dependencies.remote(cmd) for _ in range(4)]

# Wait for the result
result = ray.get(future)
print(result)

# Shut down Ray when done
ray.shutdown()
