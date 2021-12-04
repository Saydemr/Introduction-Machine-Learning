#!/bin/bash

for learning_rate in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2
do
    unbuffer python ./single_layer.py $learning_rate 2>&1 | tee ./log/single_layer_log_${learning_rate}.txt
done


# for learning_rate in 0.01 0.02 0.03 0.04 0.06 0.07 0.08 0.09 0.10 0.15 0.20
# do
#     unbuffer python ./multi_layer.py $learning_rate 2>&1 | tee ./log/multi_layer_log_${learning_rate}.txt
# done