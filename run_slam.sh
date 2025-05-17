task_name="pour_water" # or "arrange_mouse" "fold_towel" "unplug_charger"

python run_slam_pipeline.py data/dataset/${task_name}/
sleep 3s
docker stop $(docker ps -aq)
sleep 3s
