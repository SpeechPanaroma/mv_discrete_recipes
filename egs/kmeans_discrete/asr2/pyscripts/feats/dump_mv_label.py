# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_km_label.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


import argparse
import logging
import os
import sys

import joblib
import numpy as np
import torch
from ssl_feature_utils import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
    S3PRLFeatureReader,
    S3PRLFeatureReaderMultiview,
    build_data_iterator,
    format_feature_conf_str,
)

from espnet2.utils.types import str2bool
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


feature_reader_choice = dict(
    mfcc=MfccFeatureReader,
    fairseq_hubert=HubertFeatureReader,
    espnet_hubert=ESPnetHubertFeatureReader,
    s3prl_singleView=S3PRLFeatureReader,
    s3prl_multiView=S3PRLFeatureReaderMultiview,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--online_feature_extract", type=str2bool, default=False)
    parser.add_argument("--feature_conf", type=str, default=None)
    parser.add_argument("--batch_bins", type=int, default=1)
    parser.add_argument(
        "--utt2num_samples",
        type=str,
        default=None,
        help="Specify the utt2num_samples file.",
    )

    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound", "kaldi_ark"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--out_filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser


def dump_label(
    rspecifier,
    in_filetype,
    wspecifier,
    out_filetype,
    km_path,
    use_gpu,
    online_feature_extract,
    **kwargs
):
    if online_feature_extract:
        assert "feature_conf" in kwargs
        # need to wrap arguments with double-quotes for json string
        feature_conf = format_feature_conf_str(kwargs["feature_conf"])
    else:
        feature_conf = None

    multiview_model = joblib.load(km_path)

    if not online_feature_extract:
        # dumped ssl feature in kaldi ark format
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt, feat in file_reader_helper(rspecifier, in_filetype):
                lab = apply_kmeans(feat)
                writer[utt] = lab
    else:
        assert feature_conf["type"] in feature_reader_choice
        reader_class = feature_reader_choice[feature_conf["type"]]
        reader_conf = feature_conf.get("conf", dict())

        if reader_conf.get("multilayer_feature", None):
            reader_conf["multilayer_feature"] = str2bool(
                reader_conf["multilayer_feature"]
            )
        if reader_conf.get("layer", None):
            if feature_conf["type"] == "s3prl_multiview":
                reader_conf["layer"] = [int(layer) for layer in reader_conf["layer"]]
            else:
                reader_conf["layer"] = int(reader_conf["layer"])

        reader = reader_class(**reader_conf)
        iterator = build_data_iterator(
            rspecifier,
            in_filetype,
            utt2num_samples=args.utt2num_samples,
            batch_bins=kwargs.get("batch_bins", 1),
        )
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt_ids, data in iterator:
                feats, feats_lens = reader.get_feats(
                    data["speech"], data["speech_lengths"]
                )
                
                slicing_dim = 1024
                for idx, utt in enumerate(utt_ids):                   
                    multiview = [feats[idx][:,:slicing_dim], feats[idx][:,slicing_dim:]]
                    lab = multiview_model.predict(multiview)
                    writer[utt] = lab.cpu().numpy()
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))
