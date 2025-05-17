task_name="pour_water" # or "arrange_mouse" "fold_towel" "unplug_charger"

python scripts_slam_pipeline/07_generate_replay_buffer.py \
-o data/dataset/${task_name}/dataset.zarr.zip \
data/dataset/${task_name}/