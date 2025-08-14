# !/bin/bash
mkdir /home/result/fastqc

readfiles=(
    "13823_9601_184391_HVG2NBGXN_1_TGTTGACT_R1" \
    "13823_9601_184392_HVG2NBGXN_2_ACGGAACT_R1" \
    "13823_9601_184393_HVG2NBGXN_3_TCTGACAT_R1" \
    "13823_9601_184394_HVG2NBGXN_4_CGGGACGG_R1" \
    )

index_primer_seq=(
    ""
)

for readfile in ${readfiles[@]};do
    tmp=$readfile
    echo $tmp
    cutadapt -j 12 \
        -g "^GGG" \
    	-m 15 \
    	--max-n=0.1 \
    	--discard-casava \
    	-o /home/result/trimmed_GGG_${tmp}.fastq.gz \
    	/home/data/${readfile}.fastq.gz > log_cutadapt_GGG_${tmp}.txt
    
    cutadapt -j 12 \
        -a "A{10};min_overlap=5"\
        -n 2 \
    	-m 15 \
    	--max-n=0.1 \
    	--discard-casava \
    	-o /home/result/trimmed_${tmp}.fastq.gz \
    	/home/result/trimmed_GGG_${tmp}.fastq.gz > log_cutadapt_${tmp}.txt
   
done

# mapping to genome
for tmp in ${readfiles[@]};do
    readfile=$tmp
    echo $tmp
    STAR --runThreadN 12 \
        --genomeDir /home/ref/ensembl_109_hsa/STAR_genome_hsa \
        --readFilesIn /home/result/trimmed_${readfile}.fastq.gz \
        --outSAMtype BAM SortedByCoordinate  \
        --readFilesCommand zcat \
        --runDirPerm All_RWX \
        --outFileNamePrefix /home/result/STAR_genome_align2_$readfile\_ \
        --outFilterMultimapNmax 10 \
        --alignIntronMin 20 \
        --alignIntronMax 100000 \
        --outFilterMismatchNmax 10 \
        --outFilterType BySJout \
        --outFilterMismatchNoverLmax 0.04 \
        --alignEndsType EndToEnd \
        --outSAMattributes MD NH

done
