#!/usr/bin/env bash


# Check if a type was provided as a command line argument
if [[ -n "$1" ]]; then
  type="$1"
else
  type="cpu"
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

if [[ -z "$region" ]]; then
    region="us-west-2"
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$region.amazonaws.com

channel="training"

# Set the image variable based on the channel and type
image="autogluon-$channel-$type"
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
echo "region: $region"
echo "channel: $channel"
echo "type: $type"
echo "image: $image"
echo "fullname: $fullname"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.


docker build  -t ${image} . --build-arg region=${region} --build-arg type=${type} --build-arg channel=${channel}
#docker tag ${image} ${fullname}

# # If the repository doesn't exist in ECR, create it.
# aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

# if [ $? -ne 0 ]
# then
#     aws ecr create-repository --repository-name "${image}" > /dev/null
# fi

# docker push ${fullname}


############################ inference ############################
#channel="inference"
#image="autogluon-$channel-$type"
#fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
#echo "region: $region"
#echo "channel: $channel"
#echo "type: $type"
#echo "image: $image"
#echo "fullname: $fullname"
#
#docker build  -t ${image} . --build-arg region=${region} --build-arg type=${type} --build-arg channel=${channel}
##docker tag ${image} ${fullname}

# # If the repository doesn't exist in ECR, create it.
# aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

# if [ $? -ne 0 ]
# then
#     aws ecr create-repository --repository-name "${image}" > /dev/null
# fi

# docker push ${fullname}