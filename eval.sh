task_name="pour_water" # or "arrange_mouse" "fold_towel" "unplug_charger"

python eval_real.py \
--robot_config=example/eval_robots_config.yaml \
-i data/checkpoints/${task_name}/latest.ckpt \
-o data/eval_${task_name} \
--frequency 5 \
--temporal_agg -si 1