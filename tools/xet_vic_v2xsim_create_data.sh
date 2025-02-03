DATA_NAME=V2X-Sim-full-coop

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/v2xsim_data_converter/v2xsim_utils.py --root-path ./datasets/${DATA_NAME}
python tools/v2xsim_create_data.py nuscenes --root-path ./datasets/${DATA_NAME} \
       --out-dir ./data/infos/${DATA_NAME} \
       --extra-tag nuscenes \
       --version v1.0 \
       --canbus ./datasets/${DATA_NAME} \
       --coop