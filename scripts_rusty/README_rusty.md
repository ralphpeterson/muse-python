

## Optimization Steps

1. Select optimization conditions in `mp_sym_optimization.py`:
  - `room_type   = 'small'         # 'small' or 'large'`
  - `wall_type   = 'rev'           # 'rev' or 'nrev'`
  - `eval_type   = 'jk'            # evaluation type: 'sn' or 'jk'`
  - `mic_config  = 'ceiling'       # 'wall' or 'ceiling'`
  - `opt_type    = 'rot'           # optimization type: 'diag', 'vert', 'rot', or 'hght'`
2. Modify the optimization variable range in `warble_sym.sh`: 1-180 or 1-32, depending on the `opt_type`.
3. Run `warble_sym.sh` with `$ sbatch shell_scripts/warble_sym.sh`
4. Check the optimization result with `scripts_jupyternb/result_check.ipynb`

