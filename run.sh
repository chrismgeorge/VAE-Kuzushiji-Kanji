pip3 install gdown
mkdir ./pretrained_models/
cd pretrained_models/
gdown https://drive.google.com/uc?id=1Yl5Crwy08PavXpR3YPEuo_mNZ3pDIM8y
cd ..
git clone https://github.com/rois-codh/kmnist.git
mv ./download_data.py ./kmnist/
cd kmnist
python3 download_data.py
tar -xvf kkanji.tar
cd ..
python3 remove_images.py
