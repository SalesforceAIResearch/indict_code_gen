env_name=indict 

if conda info --envs | grep -q $env_name; then echo "Skip! $env_name already exists"
else conda create -n $env_name python=3.10 -y
fi
source activate $env_name

pip install --upgrade pip
pip install vllm openai 
pip install -r requirements.txt 