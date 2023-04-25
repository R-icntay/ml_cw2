#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=2:00:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Multi-task
#$ -V
#$ -wd /home/mmasramo/ml_cw2/marta

hostname

date

export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH

python3 main.py 

date
