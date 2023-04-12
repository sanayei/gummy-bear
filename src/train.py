import argparse
import os
import shutil
from pprint import pprint
import yaml
from autogluon.multimodal import MultiModalPredictor
from ag_utils.image_utils import MultimodalDF
import pandas as pd


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        # print(f"WARN: more than one file is found in {channel} directory")
        print("WARN: more than one file is found in directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


# def get_env_if_present(name):
#     result = None
#     if name in os.environ:
#         result = os.environ[name]
#     return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument(
        "--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TEST"),
    )
    # parser.add_argument(
    #     "--ag_config", type=str, default=os.environ.get("SM_CHANNEL_CONFIG")
    # )
    # parser.add_argument(
    #     "--serving_script", type=str, default=os.environ.get("SM_CHANNEL_SERVING")
    # )

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    # os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    # config_file = args.ag_config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config

    if args.n_gpus:
        config["num_gpus"] = int(args.n_gpus)

    print("Running training job with the config:")
    pprint(config)

    # ---------------------------------------------------------------- Training

    # train_file = get_input_path(args.train_dir)
    train_path = args.train_dir
    train_file = os.path.join(train_path, 'train.csv') # we assume that training file is always named train.parquet
    train_data = pd.read_csv(train_file)
    train_data = MultimodalDF(train_data)
    train_data.decode(column_name='Images', inplace=True)
    # train_file = '/opt/ml/input/data/train/train.csv'
    # train_data = train_data.sample(50)
    save_path = os.path.normpath(args.model_dir)

    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = save_path
    ag_fit_args = config["ag_fit_args"]

    predictor = MultiModalPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # --------------------------------------------------------------- Inference

    # if args.test_dir:
    #     test_path = args.test_dir
    #     # test_file = get_input_path(args.test_dir)
    #     # test_file = '/opt/ml/input/data/test/test.csv'
    #     test_file = os.path.join(test_path, 'test.csv')  # we assume that training file is always named train.parquet
    #     test_data = pd.read_csv(test_file)
    #     test_data = MultimodalDF(test_data)
    #     test_data.decode(column_name='Images', inplace=True)
    #
    #     # Predictions
    #     y_pred_proba = predictor.predict_proba(test_data)
    #     if config.get("output_prediction_format", "csv") == "parquet":
    #         y_pred_proba.to_parquet(f"{args.output_data_dir}/predictions.parquet")
    #     else:
    #         y_pred_proba.to_csv(f"{args.output_data_dir}/predictions.csv")
    #
    #     # Leaderboard
    #     if config.get("leaderboard", False):
    #         lb = predictor.leaderboard(test_data, silent=False)
    #         lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")
    #
    #     # Feature importance
    #     if config.get("feature_importance", False):
    #         feature_importance = predictor.feature_importance(test_data)
    #         feature_importance.to_csv(f"{args.output_data_dir}/feature_importance.csv")
    # else:
    #     if config.get("leaderboard", False):
    #         lb = predictor.leaderboard(silent=False)
    #         lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")
    #
    #
