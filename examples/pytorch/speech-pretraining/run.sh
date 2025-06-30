# --dataset_config_names clean clean other \
# --dataset_split_names train.100 train.360 train.500 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=4 \
accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    run_wav2vec2_pretraining_no_trainer.py \
        --dataset_name=librispeech_asr \
        --dataset_config_names clean \
        --dataset_split_names train.100 \
        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
        --output_dir="./wav2vec2-pretrained-demo" \
        --max_train_steps="200000" \
        --num_warmup_steps="32000" \
        --gradient_accumulation_steps="4" \
        --learning_rate="0.001" \
        --weight_decay="0.01" \
        --max_duration_in_seconds="20.0" \
        --min_duration_in_seconds="2.0" \
        --logging_steps="1" \
        --saving_steps="10000" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="8" \
        --adam_beta1="0.9" \
        --adam_beta2="0.98" \
        --adam_epsilon="1e-06" \
        --mask_time_prob="0.65" \
        --mask_time_length="10" \
        --local_dataset_path="/ssd1/libri/LibriSpeech" \
        --cache_dir="/workspace/speech_norm/cache"
