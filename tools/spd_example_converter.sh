DATA_NAME=$1
NO_ERR_OFFSET=$2

python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./data/infos/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side vehicle-side

python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./datasets/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side vehicle-side

python tools/spd_data_converter/map_spd_to_nuscenes.py \
    --maps-root ./datasets/${DATA_NAME}/maps \
    --save-root ./datasets/${DATA_NAME} \
    --v2x-side vehicle-side

python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./data/infos/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side infrastructure-side

python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./datasets/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side infrastructure-side

python tools/spd_data_converter/map_spd_to_nuscenes.py \
    --maps-root ./datasets/${DATA_NAME}/maps \
    --save-root ./datasets/${DATA_NAME} \
    --v2x-side infrastructure-side

python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./data/infos/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side cooperative \
    ${NO_ERR_OFFSET}

python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./datasets/${DATA_NAME} \
    --split-file ./data/split_datas/cooperative-split-data-spd-delay.json \
    --v2x-side cooperative \
    ${NO_ERR_OFFSET}

python tools/spd_data_converter/map_spd_to_nuscenes.py \
    --maps-root ./datasets/${DATA_NAME}/maps \
    --save-root ./datasets/${DATA_NAME} \
    --v2x-side cooperative