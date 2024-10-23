
# hostnameが mba.local なら uv で実行。研究室サーバーでは python で実行
if [ `hostname` = "mba.local" ]; then
    uv run python src/sn_script/download_soccernet_data.py --task v3
else
    python src/sn_script/download_soccernet_data.py --task v3
fi

