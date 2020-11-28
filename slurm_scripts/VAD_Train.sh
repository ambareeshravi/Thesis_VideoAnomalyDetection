#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:1   # Request GPU "generic resources" [--gres=gpu:2]
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=63500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]

#SBATCH --time=0-05:00      # time (DD-HH:MM)
#SBATCH --output=/scratch/destination_folder/some_name-%j.out

#SBATCH --mail-user=ambareesh.ravi@uwaterloo.ca
#SBATCH --mail-type=ALL

nvidia-smi

cp ~/projects/def-karray/a24ravi/VAD_Datasets.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/VAD_Datasets.zip
echo "[STATUS] Created data directory"

module load python/3.7.4

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index -r ~/workspace/Thesis_VideoAnomalyDetection/pip3_requirements.txt
echo "[STATUS] Python environment ready"

source /home/$USER/ENV/bin/activate
cd ~/workspace/Thesis_VideoAnomalyDetection/AutoEncoders/
python run_config.py --model_path 


rm -rf $SLURM_TMPDIR/env/






#SBATCH --job-name=some_name

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...

module load python/3.7.4
source ~/ENV/bin/activate
python ~/scratch/destination_folder/python_script_name.py
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"

#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=1-00:00
module load arch/avx512 StdEnv/2018.3
nvidia-smi


#!/bin/bash
# The following three commands allow us to take advantage of whole-node
# scheduling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
# Wall time
#SBATCH --time=12:00:00
#SBATCH --job-name=example
#SBATCH --output=$SCRATCH/output/example_jobid_%j.txt
# Emails me when job starts, ends or fails
#SBATCH --mail-user=example@gmail.com
#SBATCH --mail-type=ALL

ENV=path/to/my/env

# load any required modules
module load python/3.7.0
# activate the virtual environment
source $ENV/bin/activate

# run a training session
srun python example.py
