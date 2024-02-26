#!/bin/bash

# abort when a command returns non-zero value
set -e

cmd_list=()

# add float vector tests
metrics=("L2" "IP" "Cosine")
for metric in "${metrics[@]}"; do
    cmd_list+=( "./boost_ut_test_vector_index --metric ${metric} --use_default_params 1"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --use_default_params 1 --data_dim 1536"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --use_default_params 1 --data_dim 2048"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --use_default_params 1 --data_dim 4096"
                "./boost_ut_test_vector_index --metric ${metric} --use_default_params 1 --filter_out_mod 3"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --use_default_params 1 --scann_build_hashed_dataset_by_token 0"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --use_default_params 1 --filter_out_mod 3"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --num_data 2000 --filter_out_mod 3 --filter_keep_min 1009 --filter_keep_max 1012"
                "./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --scann_children_per_level 3_3"
                "./boost_ut_test_vector_index --index_types SCANN --num_data 200000 --metric ${metric} --check_build_canceled_sec 1"
                "SPDLOG_LEVEL=debug ./boost_ut_test_vector_index --metric ${metric} --index_types SCANN --scann_l_search_ratio 0.9"
                )
done

# add binary vector tests
bin_metrics=("Hamming" "Jaccard")
for metric in "${bin_metrics[@]}"; do
    cmd_list+=( "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryFLAT"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryIVF"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryHNSW"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryFLAT --filter_out_mod 5"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryIVF  --filter_out_mod 5"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryHNSW --filter_out_mod 5"
                )
done

for cmd in "${cmd_list[@]}"; do
    echo "## Executing $cmd"
    eval $cmd
    # clean up all the temp files
    rm -rf /tmp/test_vector_index_index_*
done

if [ -n "$STRESS_TEST" ]
then
# stress test for 10 seconds
./boost_ut_test_vector_index --index_types SCANN --metric ${metric} --stress_threads 20 --stress_sec $STRESS_TEST
fi
