ARG region
ARG type
ARG channel
FROM 763104351884.dkr.ecr.${region}.amazonaws.com/autogluon-${channel}:0.6.2-${type}-py38-ubuntu20.04

# RUN apt-get update \
#  && apt-get -y upgrade \
#  && apt-get install -y --no-install-recommends \
#  && apt-get autoremove -y \
#  && apt-get clean
#
# RUN pip3 install -U pip
# RUN pip3 install -U setuptools wheel
#
# RUN pip install -U autogluon

# COPY /.cache /root/.cache


# # install from the source
# RUN git clone https://github.com/autogluon/autogluon.git
# WORKDIR /autogluon

# RUN python3 -m pip install common/[tests]
# RUN python3 -m pip install core/[all,tests]
# RUN python3 -m pip install features/
# RUN python3 -m pip install tabular/[all,tests]
# RUN python3 -m pip install multimodal/[tests]
# RUN python3 -m pip install timeseries/[all,tests]
# RUN python3 -m pip install autogluon/

# RUN mim install mmcv-full --timeout 60
# RUN pip install --upgrade mmdet
# RUN pip install --upgrade mmocr