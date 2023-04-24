#$ -l tmem=16G,h_vmem=16G
#$ -l h_rt=2:00:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Multi-task
#$ -V
#$ -wd /home/mmasramo/ml_cw2/marta

hostname

date

python3 main.py 

date
