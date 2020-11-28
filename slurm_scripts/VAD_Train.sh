#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:1   # Request GPU "generic resources" [--gres=gpu:2]
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=63500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]

#SBATCH --time=0-02:00      # time (DD-HH:MM)

#SBATCH --mail-user=ambareesh.ravi@uwaterloo.ca
#SBATCH --mail-type=ALL

free -g
nvidia-smi

tar xf ~/projects/def-karray/a24ravi/VAD_Datasets.tar -C $SLURM_TMPDIR/
echo "[STATUS] Created data directory"

ls $SLURM_TMPDIR -a
ls $SLURM_TMPDIR/VAD_datasets/ -l | wc -l

module load python/3.7.4
source /home/$USER/ENV/bin/activate
echo "[STATUS] Python environment ready"

mkdir $SLURM_TMPDIR/Models
cd ~/workspace/Thesis_VideoAnomalyDetection/AutoEncoders/

echo "[STATUS] Starting script at `date`"
python run_config.py --model_path $SLURM_TMPDIR/Models/ --data_path $SLURM_TMPDIR/VAD_datasets/
echo "[STATUS] Script completed at `date`" 

cp run_config.py $SLURM_TMPDIR/Models/
tar cf ~/projects/def-karray/a24ravi/trained_models/`date +%d_%m_%Y_%H:%M.tar` $SLURM_TMPDIR/Models/*
echo "[STATUS] Models copied safely"