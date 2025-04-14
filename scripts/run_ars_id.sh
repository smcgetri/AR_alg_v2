#!/bin/bash
#SBATCH --job-name=ars_id
#SBATCH --account=canari
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --mem=4000
#SBATCH --output=/home/users/singet75/AR_alg_v2/logs/ars_id_test_%j.out
#SBATCH --error=/home/users/singet75/AR_alg_v2/logs/ars_id_test_%j.err

module load jaspy
export PYTHONUNBUFFERED=1
echo "Starting ARs_ID.py..."

# Run ARs_ID
/usr/bin/time -v python /home/users/singet75/AR_alg_v2/AR_alg_v2/ARs_ID.py \
    "1950-01-01_0600" "1950-12-30_1800" 6 \
    /home/users/singet75/AR_alg_v2/config_files/AR_ID_config.hjson

echo "Finished ARs_ID.py run."
