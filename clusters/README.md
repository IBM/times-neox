# Frontier

## Setup

To set up ```Times-NeoX``` in Frontier, simply clone the Github repository and copy the environment from the world-share to your working directory.

```
cd ~
mkdir times-neox
cd times-neox
git clone https://github.com/IBM/times-neox.git
cp /lustre/orion/world-shared/stf218/junqi/neox/times-env times-env
```

That's it!

# Running Scripts

To allocate jobs for training & inference, attach the following SLURM preample at the top of the script.

```
#!/bin/bash
#SBATCH -A PROJECT_CODE # E.g. csc538
#SBATCH -J JOB_NAME
#SBATCH -o path/to/job/logs/log_file_name.out # E.g. $HOME/job_logs/times-neox/%x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N NNODES
```

Some comments:

1. Remember that each Frontier node contains 8 GPUs!
1. The ```-p``` flag specifies which partition to allocate the job on, the Frontier recommended default is the "batch" partition. See the [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) for more details.
2. The maximum allocation time for a job on the "batch" partition is 2 hours. For longer job allocations, use a different partition.
3. The parameters "%x" and "%j" to the output flag "-o" refer to JOB_NAME and JOB_ID, respectively.

Once the job script has properly specified its SLURM options, make sure to properly specify the environment path and parallelism variables:

```
export PATH="$HOME/times-neox/times-env/bin:$PATH"
GPUS_PER_NODE=8
NNODES=42 # Be sure this matches the value in the preample.
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
```

The rest of the script can now be filled in with your code. 

To actually run the script, make sure the correct modules are load and the conda environment is activated.

```
module load cray-python rocm/5.4.0
conda activate ~/times-neox/times-env
```

Now we can just run our job script, e.g. ```train_script.sh```.
```
cd ~/times-neox/times-neox
sbatch train_script.sh
```

**Be sure to double-check that all parameters (especially parallelism ones) match in the job script & config file.**

Happy machine learning!
