#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature=[wavlm_large,hubert_large_ll60k]  # use model_type/layer_index or model1_type+model2_type/layer_index1/layer_index2
layer_index=[21,21]
nclusters=2000

tag="wavlm_hub_libri960_multiview"
src_lang=$(echo "${kmeans_feature}_${layer_index}_km${nclusters}" | tr -d "["  | tr -d "]" | tr "," "+")
tgt_lang="${tag}_en"
view=multiView
multilayer_feature=False

train_set="train_clean_100"
train_dev="dev_clean"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_discrete_asr_e_branchformer1.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2_mv.sh \
    --kmeans_opts "--batch_bins 4800000 --nj 32" \
    --nj 32 \
    --stage 2 \
    --stop_stage 4 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --kmeans_feature "${kmeans_feature}" \
    --layer_index "${layer_index}" \
    --nclusters "${nclusters}" \
    --ngpu 2 \
    --tag "${tag}" \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --view ${view} \
    --multilayer_feature ${multilayer_feature} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" 
