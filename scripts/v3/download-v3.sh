
# hostnameが mba.local なら uvで実行。それ以外はpythonで実行
# if [ `hostname` = "mba.local" ]; then
#     uv run python src/sn_script/download_soccernet_data.py --task v3
# else
#     python src/sn_script/download_soccernet_data.py --task v3
# fi

# visualize
uv run python src/sn_script/v3/visualize.py \
    --SoccerNet_path /Users/heste/workspace/soccernet/SoccerNet_in_lrlab \
    --save_path docs/v3/ \
    --split train \
    --tiny 3 \
    --zipped_images
