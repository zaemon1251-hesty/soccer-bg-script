
# visualize 720p
uv run python src/sn_script/v3/visualize.py \
    --SoccerNet_path /Users/heste/workspace/soccernet/SoccerNet_in_lrlab/ \
    --save_path docs/v3-720p/ \
    --split train \
    --tiny 3 \
    --resolution_height 720 \
    --resolution_width 1280 \
    --zipped_images