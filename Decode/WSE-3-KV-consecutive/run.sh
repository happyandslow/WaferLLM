#!/bin/bash

once=true
for repeat in 128 256 384 512 640 768 896 1024; do
    CONFIG="model_config/llama8B_4k_1_240.json"
    bash ./run_wse3.sh $CONFIG false 0 $repeat $once > timed_output/decode_wse3_llama8B_4k_1_P240_hostrepeat${repeat}.log 2>&1
done

once=false
# for repeat in 256 512 768 1024 1280 1536 1792 2048; do
for repeat in 128 256 384 512 640 768 896 1024; do
    CONFIG="model_config/llama8B_4k_1_240.json"
    bash ./run_wse3.sh $CONFIG false 0 $repeat $once > timed_output/decode_wse3_llama8B_4k_1_P240_devicerepeat${repeat}.log 2>&1
done


# for P in 240 256 300 360 420; do
#     CONFIG="model_config/llama8B_4k_1_$P.json"
#     bash ./run_wse3.sh $CONFIG false > output/decode_wse3_llama8B_4k_1_P${P}.log 2>&1
# done