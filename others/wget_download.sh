
cd /data3/whb/data/sharegpt4video/videos/bdd100k
wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/bdd100k/bdd100k_videos.zip
unzip bdd100k_videos.zip
rm -rf bdd100k_videos.zip

cd /data3/whb/data/sharegpt4video/videos/ego4d
for i in {1..4}
do
    wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/ego4d/ego4d_videos_$i.zip
    unzip ego4d_videos_$i.zip
    rm -rf ego4d_videos_$i.zip
done

cd /data3/whb/data/sharegpt4video/videos/mixit
wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/mixit/mixkit_videos.zip
unzip mixkit_videos.zip
rm -rf mixkit_videos.zip

cd /data3/whb/data/sharegpt4video/videos/panda
for i in {1..21}
do
    wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/panda/panda_videos_$i.zip
    unzip panda_videos_$i.zip
    rm -rf panda_videos_$i.zip
done

cd /data3/whb/data/sharegpt4video/videos/pexels
for i in {1..43}
do
    wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/pexels/pexels_videos_$i.zip
    unzip pexels_videos_$i.zip
    rm -rf pexels_videos_$i.zip
done

cd /data3/whb/data/sharegpt4video/videos/pixabay
for i in {1..14}
do
    wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/pixabay/pixabay_videos_$i.zip
    unzip pixabay_videos_$i.zip
    rm -rf pixabay_videos_$i.zip
done