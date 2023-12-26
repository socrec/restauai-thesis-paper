from sagemaker.tensorflow import TensorFlowModel
import sagemaker

sess = sagemaker.Session()
role = 'arn:aws:iam::321977351608:role/service-role/AmazonSageMaker-ExecutionRole-20210912T114289'

model = TensorFlowModel(
    model_data='s3://sagemaker-ap-southeast-1-321977351608/sagemaker-tensorflow-220306-1120-013-0f744799/output/model.tar.gz',
    role=role,
    framework_version='1.12'
)

tf_predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.4xlarge')
tf_endpoint_name = tf_predictor.endpoint_name