#!/bin/bash
DIR=/mnt/raid5/data/roboset/v0.3/*

for task in $DIR
do
    taskname="$(basename $task)"
    if [ "$taskname" = "release" ]; then
        echo "Skipping release"
    else
        echo "Starting data checks and videos for $taskname"
        nohup $CONDA_PREFIX/bin/python data_checker.py -p "$task"/ > logs/"$taskname"_data.txt &
        nohup $CONDA_PREFIX/bin/python getvideo.py -p "$task"/ >logs/"$taskname"_video.txt &    
    fi
done
# echo "kill all server"
# tmux kill-session -t data_check
# tmux kill-session -t vidgen
# sleep 2

# echo "Starting data checks and videos for $DIR"
# nohup $CONDA_PREFIX/bin/python data_checker.py -p $DIR > logs/test_data.txt &
# nohup $CONDA_PREFIX/bin/python getvideo.py -p $DIR >logs/test_video.txt &

# tmux new -s data_check_4 -d "$CONDA_PREFIX/bin/python data_checker.py -p $DIR"
# tmux new -s vidgen_4 -d "$CONDA_PREFIX/bin/python getvideo.py -p $DIR"