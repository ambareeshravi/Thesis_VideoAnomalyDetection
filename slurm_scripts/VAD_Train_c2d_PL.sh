#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:1   # Request GPU "generic resources" [--gres=gpu:2]
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=63G        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]

#SBATCH --time=0-6:00      # time (DD-HH:MM)
#SBATCH --output=~/projects/def-karray/a24ravi/slurm_outputs/PL_C2D_%u-%x-%j.txt

#SBATCH --mail-user=ambareesh.ravi@uwaterloo.ca
#SBATCH --mail-type=ALL

version=PL_C2D_$(date +%d_%m_%Y_%H_%M)
dataset=VAD_Datasets
project_dir=~/projects/def-karray/a24ravi/
version_path=$SLURM_TMPDIR/$version/
mkdir -p $version_path
echo "[STATUS] Created version: $version"

free -g
nvidia-smi

tar xf ${project_dir}${dataset}.tar -C $SLURM_TMPDIR/
echo "[STATUS] Created data directory"

module load python/3.7.4
source /home/$USER/ENV/bin/activate
echo "[STATUS] Python environment ready"

cd ~/workspace/Thesis_VideoAnomalyDetection/AutoEncoders/

echo "[STATUS] Starting script at `date`"
cp run_config_c2d_PL.py $version_path
python run_config_c2d_PL.py --model_path $version_path --data_path $SLURM_TMPDIR/ --nodes 1
echo "[STATUS] Script completed at `date`" 
 
cp ${project_dir}slurm_outputs/PL_C2D_%u-%x-%j.txt $version_path
tar -cjf ${project_dir}trained_models/${version}.tar -C $version_path $(ls $version_path)
echo "[STATUS] Models copied safely"

deactivate
echo "[STATUS] Deactivate python environment. EXITING ..."