
# hostnameが mba.local なら uvで実行。それ以外はpythonで実行
if [ `hostname` = "mba.local" ]; then
    uv run python src/sn_script/download_soccernet_data.py --task v3
else
    python src/sn_script/download_soccernet_data.py --task v3
fi