# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging
import os
import sys
import torch

import joblib
import numpy as np
# from mvlearnPanorama.Cluster.mv_kmeans_minibatch import mvKMeansMinibatch
from mv_kmeans_minibatch import mvKMeansMinibatch
from espnet.utils.cli_readers import file_reader_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_Multiview_kmeans")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=300, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--n_init", default=10, type=int)
    parser.add_argument("--n_jobs", default=10, type=int)
    parser.add_argument("--batch_size", default=100000, type=int)

    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier",
        type=str,
        nargs="+",
        help="Read specifier for feats. e.g. ark:some.ark",
    )
    return parser

def save_features(features, file_path):
    torch.save(features, file_path)

def concat_feature_shards(save_dir, chunk_prefix, device):
    multiview = torch.zeros(0, 1024).to(device)  # Placeholder

    chunk_files = sorted([f for f in os.listdir(save_dir) if f.startswith(chunk_prefix)])
    for chunk_file in chunk_files:
        chunk_path = os.path.join(save_dir, chunk_file)
        chunk_data = torch.load(chunk_path).to(device)
        multiview = torch.cat((multiview, chunk_data), 0)
        print(f"Loaded and concatenated {chunk_file}")

    return multiview

def load_feature_shard_minibatch_save(rspecifiers, in_filetype, device1, device2, save_dir, chunk_size=5000):
    multiviews = [torch.zeros(0, 1024).to(device1), torch.zeros(0, 1024).to(device2)]  # Placeholder
    chunk_counter = 0
    for rspecifier in rspecifiers:
        utt_count = 0
        for utt, feat in file_reader_helper(rspecifier, in_filetype):
            logging.info("Features loaded from " + str(utt_count))
            utt_count += 1
            slicing_dim = int(len(feat[0]) / 2)
            mv_1 = torch.tensor(feat[:, :slicing_dim]).to(device1)
            mv_2 = torch.tensor(feat[:, slicing_dim:]).to(device2)

            multiviews[0] = torch.cat((multiviews[0], mv_1), 0)
            multiviews[1] = torch.cat((multiviews[1], mv_2), 0)

            if utt_count % chunk_size == 0:
                save_features(multiviews[0], os.path.join(save_dir, f'multiview_1_chunk_{chunk_counter}.pt'))
                save_features(multiviews[1], os.path.join(save_dir, f'multiview_2_chunk_{chunk_counter}.pt'))
                multiviews = [torch.zeros(0, 1024).to(device1), torch.zeros(0, 1024).to(device2)]  # Reset placeholders
                chunk_counter += 1

        logger.info("Multiview 1 device cuda status: " + str(multiviews[0].is_cuda))
        logger.info("Multiview 2 device cuda status: " + str(multiviews[1].is_cuda))
    
    # Save any remaining data after the loop
    if multiviews[0].size(0) > 0:
        save_features(multiviews[0], os.path.join(save_dir, f'multiview_1_chunk_{chunk_counter}.pt'))
        save_features(multiviews[1], os.path.join(save_dir, f'multiview_2_chunk_{chunk_counter}.pt'))

    return multiviews

def load_feature_shard(rspecifiers, in_filetype,device1,device2):
    multiviews= [torch.zeros(0,1024).to(device1),torch.zeros(0,1024).to(device2)] #Place holder
    for rspecifier in rspecifiers:
        utt_count = 0
        for utt, feat in file_reader_helper(rspecifier, in_filetype):
            logging.info("Features loaded from "+str(utt_count))
            utt_count = utt_count + 1
            slicing_dim = int(len(feat[0])/2)
            mv_1 = torch.tensor(feat[:,:slicing_dim]).to(device1)
            mv_2 = torch.tensor(feat[:,slicing_dim:]).to(device2)

            multiviews[0] = torch.cat((multiviews[0],mv_1),0)
            multiviews[1] = torch.cat((multiviews[1],mv_2),0)

        logger.info("Multiview 1 device cuda status: "+str(multiviews[0].is_cuda))
        logger.info("Multiview 2 device cuda status: "+str(multiviews[1].is_cuda))
    return multiviews

def load_feature(rspecifiers,in_filetype,device1,device2,save_dir="multiview_features"):
    if not isinstance(rspecifiers, list):
        rspecifiers = [rspecifiers]
    ##if using minibatch saving
    #multiviews = []
    #multiviews[0] = concat_feature_shards(save_dir='Librispeech_features', chunk_prefix='multiview_1_chunk_', device=device1)
    #multiviews[1] = concat_feature_shards(save_dir='Librispeech_features', chunk_prefix='multiview_2_chunk_', device=device2)
    #return multiviews 
    return load_feature_shard(rspecifiers, in_filetype,device1,device2)

def learn_kmeans(
    rspecifier,
    in_filetype,
    km_path,
    n_clusters,
    seed,
    init,
    max_iter,
    tol,
    n_init,
    patience,
    n_jobs,
    batch_size,
):
    np.random.seed(seed)
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')
    feat = load_feature(rspecifier, in_filetype,device1,device2)
#    feat = torch.load("feature.pt")
#    torch.save(feat,"librispeech_feature.pt")
    logging.info("Load feature complete")

    Multiview_Model = mvKMeansMinibatch(
        n_clusters,
        max_iter,
        patience,
        tol,
        init,
        batch_size,
        n_jobs,
        n_init)

    logging.info("Multiview model fit start")
    Multiview_Model.fit(feat)
    logging.info("Multiview model fit end")
    joblib.dump(Multiview_Model, km_path)
    logging.info("Dump complete")

    # inertia = -multiview_model_model.score(feat) / len(feat)
    # logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
