# init vd-pcr
git clone https://github.com/HKUST-KnowComp/VD-PCR VD-PCR
cd VD-PCR
conda env create -f environment.yml
conda activate vd-pcr

#if not installed apex look down
cd ../apex/
pip install -v --disable-pip-version-check --no-cache-dir ./
pip install gdown

# check the mounts
df -h
blkid /dev/sda1

# if already mounted
sudo unmount /dev/sda1
# remount
sudo mount -t ntfs /dev/sda1 disk OR sudo mount -t ext4 /dev/sda1 disk
sudo chmod a+rwx disk
ls -l disk

# download chekpoints
mkdir -p disk/logs/joint
mkdir -p disk/logs/vonly

gdown 1omwYw-YKjqAHmPA1YBqFVH18giiKkK5Z -O disk/logs/joint/MB-JC/
gdown 1M54CtnLMcPBl0I0iKOnamWpM4-jf3VXM -O disk/logs/vonly/MB-JC-HP-crf_cap-trainval/

# download visual vectors (around 45 min)
mkdir -p disk/data/visdial/visdial_img_feat.lmdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/data.mdb -O disk/data/visdial/visdial_img_feat.lmdb/data.mdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/lock.mdb -O disk/data/visdial/visdial_img_feat.lmdb/lock.mdb

# download visdial dialog splits
gdown https://drive.google.com/drive/folders/1w5L_i8h9h32dCCZpOJDsur4TIYLOtS8F -O /data/all --folder


# download clevr images
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O disk/clevr

# change config
vim config/vonly.conf

# run Phase 1
GPU=0 python main.py --model vonly/MB-JC_predict --mode predict
GPU=0 python main.py --model conly/MB-JC_predict --mode predict

python postprocessing/merge_predictions.py --mode phase1 --model MB-JC_predict

# Phase 2
python preprocessing/extract_relevant_history.py --include_cap --save_name crf_cap
GPU=0 python main.py --model vonly/MB-JC-HP-crf_cap-test --mode predict
python ensemble.py --exp convert --mode predict

# for excecuting in the background use
tmux a -t 0

# to copy file from server to local run
scp -P 62068 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/CLEVR_v1.0/images/val/CLEVR_val_000000.png /home/kuzya/Desktop/uni/GLP
scp -P 60121 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/VD-PCR/logs/conly/MB-JC_eval/test_coref_prediction.jsonlines /home/kuzya/Desktop/uni/GLP/VD_PCR_predictions
# to copy file from local t0 server run
scp -P 60121 clevr/CLEVR_VD_VAL_VISDIAL_1_dialog_per_1000_pictures.json disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/datasets/clevr
../disk/datasets/
# vilbert install
inspo from https://naserian-elahe.medium.com/vilbert-a-model-for-learning-joint-representations-of-image-content-and-natural-language-47f56a313a79
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
mamba create -n vilbert-mt python=3.6
mamba activate vilbert-mt
git clone https://github.com/KAndHisC/vilbert-multi-task.git
vim vilbert-multi-task/requirements.txt (delete numpy)
pip install -r vilbert-multi-task/requirements.txt
pip install ninja yacs cython matplotlib
mamba install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd maskrcnn-benchmark
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
export PATH=/usr/local/cuda-11.6/bin:$PATH
python setup.py build develop
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..

cd disk
mkdir data
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
cd ../../vilbert-multi-task
im.convert('RGB')

python script/extract_features.py --model_file ../disk/data/detectron_model.pth --config_file ../disk/data/detectron_config.yaml --image_dir ../disk/CLEVR_v1.0/images/val --output_folder ../disk/CLEVR_v1.0/image_features

python script/convert_to_lmdb.py --features_dir ../disk/CLEVR_v1.0/image_features_1000 --lmdb_file ../disk/CLEVR_v1.0/image_features_1000.lmdb
scp -P 60121 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/CLEVR_v1.0/image_features_1000.lmdb/data.mdb /home/kuzya/Desktop/uni/GLP/clevr/image_features_1000.lmdb
scp -P 60121 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/CLEVR_v1.0/image_features_1000.lmdb/lock.mdb /home/kuzya/Desktop/uni/GLP/clevr/image_features_1000.lmdb

# debugging
from utils.image_features_reader import  ImageFeaturesH5Reader
ifr = ImageFeaturesH5Reader('../disk/CLEVR_v1.0/image_features_1000.lmdb') 
ifr['CLEVR_val_013021']

import pickle
import base64
image_id = str('CLEVR_val_013021').encode()
index = ifr._image_ids.index(image_id)
txn = ifr.env.begin(write=False)
item = pickle.loads(txn.get(image_id))
item["features"]



To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.6/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall
Logfile is /var/log/cuda-installer.log
