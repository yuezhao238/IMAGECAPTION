conda create -n imagecaption python=3.10
conda init
conda activate imagecaption

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install aac_metrics==0.4.6
pip install nltk==3.5
pip install rouge==1.0.1