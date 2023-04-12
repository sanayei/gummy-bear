from src.ag_utils.ag_model import (
    AutoGluonSagemakerEstimator,
)
import sagemaker
from sagemaker.local import LocalSession
from sagemaker import utils
from src.ag_utils.image_utils import MultimodalDF
import pandas as pd
import yaml
import boto3

with open('src/config.yaml') as f:
    config = yaml.safe_load(f)  # AutoGluon-specific config
SETTINGS = config.get('settings')
INSTANCE_TYPE = SETTINGS.get('instance_type')
MODEL_PATH = SETTINGS.get(f'model_path')
MODEL_DATA = '{}/model.tar.gz'.format(MODEL_PATH)
REGION = SETTINGS.get('region')
DEFAULT_BUCKET = SETTINGS.get('default_bucket')
ROLE = SETTINGS.get('role')
OUTPUT_PATH = SETTINGS.get('output_path')
TRAIN_PATH = SETTINGS.get('train_path')
TEST_PATH = SETTINGS.get('test_path')
BASE_JOB_NAME = SETTINGS.get('base_job_name')

if SETTINGS.get('instance_type') == 'local':
    session = LocalSession()
    session.config = {'local': {'local_code': True}}
else:
    boto_session = boto3.Session(region_name=REGION)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=DEFAULT_BUCKET,
    )

timestamp = utils.sagemaker_timestamp()
output_path = OUTPUT_PATH
train_data = '{}/train.csv'.format(TRAIN_PATH)
test_data = '{}/test.csv'.format(TEST_PATH)
# inference_script = "file://scripts/serve.py"
# Provide inference script so the script repacking is not needed later
# See more here: https://docs.aws.amazon.com/sagemaker/latest/dg/mlopsfaq.html
# Q. Why do I see a repack step in my SageMaker pipeline?
job_name = utils.unique_name_from_base(BASE_JOB_NAME)


def download_and_process_data():
    download_dir = './temp'
    zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
    from autogluon.core.utils.loaders import load_zip
    load_zip.unzip(zip_file, unzip_dir=download_dir)
    dataset_path = download_dir + '/petfinder_for_tutorial'
    train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
    test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
    test_data = MultimodalDF(test_data)
    train_data = MultimodalDF(train_data)
    test_data.load_images(image_col='Images', image_dir=dataset_path)
    train_data.load_images(image_col='Images', image_dir=dataset_path)
    test_data.encode(column_name='Images', inplace=True)
    train_data.encode(column_name='Images', inplace=True)
    test_data.to_csv('./input/test.csv')
    train_data.to_csv('./input/train.csv')


if __name__ == '__main__':
    # download_and_process_data()
    # train

    ag = AutoGluonSagemakerEstimator(
        role=ROLE,
        entry_point="train.py",
        source_dir='src',
        custom_image_uri='autogluon-training-cpu:latest',
        region=REGION,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        # sagemaker_session=session,
        # framework_version="0.6",
        # py_version="py38",
        base_job_name=job_name,
        disable_profiler=True,
        debugger_hook_config=False,
        output_path=MODEL_PATH
    )
    #
    ag.fit(
        {
            "train": train_data,
            "test": test_data,
            "serving": "file://src/serve.py",
        },
        job_name=job_name,
    )

    # Endpoint Deployment

    # model = AutoGluonNonRepackInferenceModel(
    #     model_data=MODEL_DATA,
    #     role=role,
    #     region=region,
    #     framework_version="0.6",
    #     py_version="py38",
    #     instance_type=INSTANCE_TYPE,
    #     source_dir="scripts",
    #     entry_point="tabular_serve.py",
    #     custom_image_uri='autogluon-cpu:latest'
    # )
    # model.deploy(initial_instance_count=1,
    #              serializer=CSVSerializer(),
    #              instance_type=INSTANCE_TYPE,
    #              session=session)
    #
    # predictor = AutoGluonRealtimePredictor(model.endpoint_name)
    # df = pd.read_csv("data/test.csv")
    # data = df[:100]
    # preds = predictor.predict(data.drop(columns="class"))
    # print(preds)

    # model = AutoGluonSagemakerInferenceModel(
    #     model_data=MODEL_DATA,
    #     # custom_image_uri='autogluon-cpu:latest',
    #     custom_image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:0.6.2-cpu-py38',
    #     role=role,
    #     region=region,
    #     framework_version="0.6",
    #     py_version="py38",
    #     instance_type=INSTANCE_TYPE,
    #     entry_point="tabular_serve-batch.py",
    #     source_dir="scripts",
    #     predictor_cls=AutoGluonBatchPredictor,
    #     env={
    #     'MY_ENV_VAR': 'my_value',
    #     'ANOTHER_ENV_VAR': 'another_value'
    # }
    # )
    #
    #
    # transformer = model.transformer(
    #     instance_count=1,
    #     instance_type=INSTANCE_TYPE,
    #     strategy="MultiRecord",
    #     max_payload=6,
    #     max_concurrent_transforms=1,
    #     output_path=output_path,
    #     accept="application/json",
    #     assemble_with="Line",
    # )
    # test_input = 'file://data/test_no_header.csv'
    #
    # transformer.transform(
    #     test_input,
    #     input_filter="$[:14]",  # filter-out target variable
    #     split_type="Line",
    #     content_type="text/csv",
    #     output_filter="$['class']",  # keep only prediction class in the output
    # )
    #
    # transformer.wait()
