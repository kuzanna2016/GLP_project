# Mount additional temporary space
### Check the mounts
```bash
df -h
blkid /dev/sda1
```
### remount
```bash
sudo mount -t ntfs /dev/sda1 disk #OR sudo mount -t ext4 /dev/sda1 disk
sudo chmod a+rwx disk
ls -l disk
```

# ViLBERT
To get image features we need to install ViLBERT model  
Mostly taken from [this tutorial](https://naserian-elahe.medium.com/vilbert-a-model-for-learning-joint-representations-of-image-content-and-natural-language-47f56a313a79)
### Install Mamba
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
mamba create -n vilbert-mt python=3.6
mamba activate vilbert-mt
```
### Install requirements for ViLBERT
```bash
git clone https://github.com/KAndHisC/vilbert-multi-task.git
vim vilbert-multi-task/requirements.txt #(delete numpy)
pip install -r vilbert-multi-task/requirements.txt
pip install ninja yacs cython matplotlib
mamba install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```
### Install requirements for internal model
```bash
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd maskrcnn-benchmark
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
export PATH=/usr/local/cuda-11.6/bin:$PATH
python setup.py build develop
cd ..
```
### Instal Apex (be carefull with the version)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
```
### Download model checkpoints
```bash
cd disk
mkdir data
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
cd ../../
```
### Download CLEVR images (10 min)
```bash
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O disk/clevr.zip
unzip -q disk/clevr.zip
```
### Run the model
```bash
cd vilbert-multi-task
python script/extract_features.py --model_file ../disk/data/detectron_model.pth --config_file ../disk/data/detectron_config.yaml --image_dir ../disk/CLEVR_v1.0/images/val --output_folder ../disk/CLEVR_v1.0/image_features
python script/convert_to_lmdb.py --features_dir ../disk/CLEVR_v1.0/image_features_1000 --lmdb_file ../disk/CLEVR_v1.0/image_features_1000.lmdb
scp -P 60121 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/CLEVR_v1.0/image_features_1000.lmdb/data.mdb /home/kuzya/Desktop/uni/GLP/clevr/image_features_1000.lmdb
cd ../
```

# VD-PCR
### Init vd-pcr
```bash
git clone https://github.com/HKUST-KnowComp/VD-PCR VD-PCR
cd VD-PCR
mamba env create -f environment.yml
mamba activate vd-pcr
pip install gdown
```

### If not installed, install apex (be careful with the version)
```bash
cd ../apex/
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../
```

### Download chekpoints
```bash
mkdir -p disk/logs/joint
mkdir -p disk/logs/vonly

gdown 1omwYw-YKjqAHmPA1YBqFVH18giiKkK5Z -O disk/logs/joint/MB-JC/
gdown 1M54CtnLMcPBl0I0iKOnamWpM4-jf3VXM -O disk/logs/vonly/MB-JC-HP-crf_cap-trainval/
```
### [Visdial] Download visual vectors (around 45 min) 
```bash
mkdir -p disk/data/visdial/visdial_img_feat.lmdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/data.mdb -O disk/data/visdial/visdial_img_feat.lmdb/data.mdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/lock.mdb -O disk/data/visdial/visdial_img_feat.lmdb/lock.mdb
```
### [CLEVR-VD] Download visual vectors (around 10 min)
```bash
gdown 1nqj2S6ErUDLEW3jZYYm84J_GaKcEWnl5 -O disk/data/visdial/
unzip -q image_features_1000.lmdb.zip
```
### [Visdial] Download visdial dialog splits
```bash
gdown https://drive.google.com/drive/folders/1w5L_i8h9h32dCCZpOJDsur4TIYLOtS8F -O disk/data/all --folder
```
### [CLEVR-VD] Download CLEVR-VD
```bash
gdown 1XdqIyNNYkKt_l7_QLdaPs32iwCkHcM63 -O disk/data/clevr/
gdown 1LiC_YXqPq0qfhEh_0KU2JX17rRNnhNwm -O disk/data/clevr/
```
### Download configs
```
scp -P 60121 clevr/config/conly.conf disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/VD-PCR/config/
scp -P 60121 clevr/config/ensemble.conf disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/VD-PCR/config/
scp -P 60121 clevr/config/joint.conf disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/VD-PCR/config/
scp -P 60121 clevr/config/vonly.conf disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/VD-PCR/config/
```

### Run Phase 1
```bash
cd VD-PCR
GPU=0 python main.py --model joint/MB-JC --mode eval

GPU=0 python main.py --model vonly/MB-JC_predict --mode predict
GPU=0 python main.py --model conly/MB-JC_predict --mode predict
GPU=0 python main.py --model conly/MB-JC_eval --mode eval
GPU=0 python main.py --model conly/MB-JC_eval --mode predict
GPU=0 python main.py --model vonly/MB-JC_predict --mode predict

python postprocessing/merge_predictions.py --mode phase1 --model MB-JC_predict
```
### Phase 2
```bash
python preprocessing/extract_relevant_history.py --include_cap --save_name crf_cap
GPU=0 python main.py --model vonly/MB-JC-HP-crf_cap-test --mode predict
python ensemble.py --exp convert --mode predict
```
# Misc
### For excecuting in the background use
```tmux a -t 0```

### To copy file from server to local run
```bash
scp -P 62068 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/CLEVR_v1.0/images/val/CLEVR_val_000000.png /home/kuzya/Desktop/uni/GLP
scp -P 60121 disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/2801session.txt /home/kuzya/Desktop/uni/GLP/
```

### to copy file from local to server run
```bash
scp -P 60121 clevr/CLEVR_VD_VAL_VISDIAL_1_dialog_per_1000_pictures.json disi@ml-lab-2a8eaddc-f375-4ce9-914c-c69e61f6f4ec.westeurope.cloudapp.azure.com:~/disk/datasets/clevr
```
