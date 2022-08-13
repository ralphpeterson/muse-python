#!/bin/bash
#SBATCH -p ccn
#SBATCH -t 1-12:00:00   # Runtime in D-HH:MM
#SBATCH -N1  # request 1 node
#SBATCH -c 128
#SBATCH --mem=64000mb
#SBATCH --output=logs/logs_sym/sml_diag_%j.log

# SBATCH --output=logs/logs_sym_diag/sml_vert_diag_%j.log
# SBATCH -n 1 # --exclusive #(this is pre-configured, no need to have this) -n for a number of tasks
# SBATCH --constraint=skylake
hostname;date;pwd;

source ~/.bashrc
source ~/venv/bin/activate

# echo "test 8 mics"
# srun python ~/scripts/mp_8_mic_test.py
# srun python ~/scripts/mp_sym.py --alpha 0.03;
# srun python ~/scripts/mp_sym_vert_diag.py
# srun python ~/scripts/mp_sym_diag.py --alpha 0.2

# -*-*-*-*-*-*- Optimal symmetric config search: Diagonal/Vertical -*-*-*-*-*-*-
# for i in {1..32}; do
#     echo "Running: $i..."
#     # echo "alpha = $((0.03*$i))"
#     srun python ~/scripts/mp_sym_optimization.py --opt_var $i;
#     # srun python ~/scripts/mp_sym_diag.py --alpha $i;
#     # srun python ~/scripts/mp_sym_vert.py --alpha $i;
# done
# srun python ~/scripts/mp_sym_optimization.py --opt_var 20

# -*-*-*-*-*-*- Optimal symmetric config search: Rotational -*-*-*-*-*-*-
for i in {1..180}; do
    # echo "Running: $i..."
    srun python ~/scripts/mp_sym_optimization.py --opt_var $i;
    # echo "$i is done."
done

# srun python ~/scripts/mp_wbl.py --idxs $m1 $m2 27;
# srun python ~/scripts/mp_wbl.py --idxs $m1 $m2 $m3 $m4 500;

# rng=$1 # input argument
# echo "rng = $rng"
# echo "from $(($(($rng-1))*100)) to $((100*$rng))"
# for mx in $(eval echo {$(($(($rng-1))*100))..$((100*$rng))}); do
#     echo "evaluating: $mx"
#     date "+%Y-%m-%d %H:%M:%S"
#     srun python ~/scripts/mp_wbl.py --idxs $m1 $m2 $m3 $m4 $mx;
#     echo "$mx is done"
# done

# for mx in {1504..1599}; do
#     echo "evaluating: $mx";
#     date "+%Y-%m-%d %H:%M:%S"
#     srun python ~/scripts/mp_wbl.py --idxs $m1 $m2 $m3 $m4 $mx;
#     echo "$mx is done"
# done

# for i in {0..10}; do
#     for j in $(eval echo {$(($i+1))..11}); do
#         # echo "i=$i, j=$j"
#         srun python ~/scripts/mp_wbl.py --idxs m1 m2 $i;
#     done
# done

date;