# !/bin/bash

curdir=`echo $PWD`
datadir="/home/saori/Git/myRiboSeq_res/data/published/Lee2012/fastq"

docker run -it \
    --name tmp \
    -v $curdir:/home/result \
    -v $datadir:/home/data \
    -v /home/saori/Git/myRiboSeq_res/ref:/home/ref \
    saori/riboseq:03
docker rm tmp
