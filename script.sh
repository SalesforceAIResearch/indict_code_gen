model=llama3-8b-instruct 
task=$1  

strategy=indict_llama 

echo " Experiment Config: $task $model $strategy " 

source activate indict

output_path=${task}_${model}
for i in {1..3}
do
    echo "Generation round # $i"
    if [ $i -eq 1 ]; then
        suffix=_round${i}
        python run.py --model $model \
            --task $task --strategy $strategy \
            --suffix $suffix \
            --debug 
    else
        prev_trial_path=${output_path}/${strategy}_round$(($i - 1))/
        suffix=_round${i}
        echo "Prior trial path: $prev_trial_path"
        python run.py --model $model \
            --task $task --strategy $strategy \
            --prev_trial $prev_trial_path  \
            --suffix $suffix \
            --debug
    fi    
done