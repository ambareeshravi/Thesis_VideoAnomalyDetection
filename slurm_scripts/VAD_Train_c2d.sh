#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:2   # Request GPU "generic resources" [--gres=gpu:2]
#SBATCH --cpus-per-task=44  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=120G        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]

#SBATCH --time=0-4:00      # time (DD-HH:MM)

#SBATCH --mail-user=ambareesh.ravi@uwaterloo.ca
#SBATCH --mail-type=ALL

free -g
nvidia-smi

tar xf ~/projects/def-karray/a24ravi/VAD_Datasets.tar -C $SLURM_TMPDIR/
echo "[STATUS] Created data directory"

module load python/3.7.4
source /home/$USER/ENV/bin/activate
echo "[STATUS] Python environment ready"

mkdir $SLURM_TMPDIR/Models
cd ~/workspace/Thesis_VideoAnomalyDetection/AutoEncoders/

echo "[STATUS] Starting script at `date`"
python run_config_c2d.py --model_path $SLURM_TMPDIR/Models/ --data_path $SLURM_TMPDIR/
echo "[STATUS] Script completed at `date`" 

 
for d in $SLURM_TMPDIR/Models/*/; do cp run_config_c2d.py "$d"; done

tar cf ~/projects/def-karray/a24ravi/trained_models/`date +%d_%m_%Y_%H.tar` $SLURM_TMPDIR/Models/*
echo "[STATUS] Models copied safely"

deactivate
echo "[STATUS] Deactivate python environment"
