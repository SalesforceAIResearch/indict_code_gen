model=CohereForAI/c4ai-command-r-v01 
max_len=10000 #10k-20k tokens to be safe  

source activate indict 

pip install --upgrade pydantic 
pip install flash_attn 

export HF_TOKEN=<HUGGING FACE TOKEN HERE>

export RAY_memory_monitor_refresh_ms=0; 
export CUDA_VISIBLE_DEVICES=0,1,2,3
size=4

python -u -m vllm.entrypoints.openai.api_server --model $model --tensor-parallel-size $size --max-model-len $max_len