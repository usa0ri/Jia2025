# !/bin/bash

readfiles=(
    "14189_9601_204504_HN2HLAFX5_1_TCTGACAT_R1" \
    "14189_9601_204505_HN2HLAFX5_2_CGGGACGG_R1" \
    "14189_9601_204506_HN2HLAFX5_3_GTGCGGAC_R1" \
    "14189_9601_204507_HN2HLAFX5_4_CGTTTCAC_R1" \
    "14189_9601_204508_HN2HLAFX5_5_AAGGCCAC_R1" \
    "14189_9601_204509_HN2HLAFX5_6_TCCGAAAC_R1"
    )

# mkdir /home/result/fastqc
# for readfile in ${readfiles[@]};do
#     fastqc -t 12 -f fastq -o /home/result/fastqc /home/data/${readfile}.fastq.gz
# done

for readfile in ${readfiles[@]};do
    tmp=${readfile%%_R1_001*}
    echo $tmp
    cutadapt -j 12 \
    	-g "^GGG" -a "A{10}" -n 2\
    	-m 12 \
    	--max-n=0.1 \
    	--discard-casava \
    	-o /home/result/trimmed_${tmp}.fastq.gz \
    	/home/data/${readfile}.fastq.gz > log_cutadapt_${tmp}.txt
done

for tmp in ${readfiles[@]};do
    readfile=${tmp%%_R1_001*}
    echo $tmp
    bowtie --quiet \
		-q \
		-v 0 \
		--norc \
		-p 12 \
		-S \
		--sam-nohead \
		--un /home/result/rRNA_left_${readfile}.fastq \
		-q /home/ref/Homo_sapiens_yuanhui/ref_rRNA/rRNA_NCBI_ENS_merged \
		<(zcat /home/result/trimmed_${readfile}.fastq.gz) \
		| awk 'BEGIN{FS="\t"}{if($2==0){print}}' \
		>> /home/result/rRNA_align_${readfile}.fastq
    gzip /home/result/rRNA_left_${readfile}.fastq
    gzip /home/result/rRNA_align_${readfile}.fastq
done

for tmp in ${readfiles[@]};do
    readfile=${tmp%%_R1_001*}
    echo $tmp
    STAR --runThreadN 12 \
        --genomeDir /home/ref/Homo_sapiens_109_saori/STAR_ref \
        --readFilesIn /home/result/rRNA_left_${readfile}.fastq.gz \
        --outSAMtype BAM SortedByCoordinate  \
        --readFilesCommand zcat \
        --runDirPerm All_RWX \
        --outFileNamePrefix /home/result/STAR_align_$readfile\_ \
        --outSAMattributes All \
        --outFilterScoreMinOverLread 0 \
        --outFilterMatchNminOverLread 0 \
        --outBAMsortingBinsN 200
done

##############################
for tmp in ${readfiles[@]};do
    readfile=${tmp%%_R1_001*}
    file=/home/result/STAR_align_${readfile}_Aligned.sortedByCoord.out.bam
    samtools view -H $file > /home/result/header
    samtools view $file \
        | grep -P "^\S+\s0\s" \
        | grep -P "NH:i:1\b" \
        | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
        | cat /home/result/header -| samtools view -bS ->/home/result/uniq_STAR_align_${readfile}.bam
    samtools index /home/result/uniq_STAR_align_${readfile}.bam
    # primary
    samtools view $file \
        | grep -P "^\S+\s0\s" \
        | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
        | cat /home/result/header -| samtools view -bS ->/home/result/primary_STAR_align_${readfile}.bam
    samtools index /home/result/primary_STAR_align_${readfile}.bam
done

for tmp in ${readfiles[@]};do
    readfile=${tmp%%_R1_001*}
    file=/home/result/STAR_align_${readfile}_Aligned.sortedByCoord.out.bam
    samtools view -H $file > /home/result/header
    samtools view $file \
        | grep -P "^\S+\s16\s" \
        | grep -P "NH:i:1\b" \
        | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
        | cat /home/result/header -| samtools view -bS ->/home/result/uniq_rev_STAR_align_${readfile}.bam
    samtools index /home/result/uniq_rev_STAR_align_${readfile}.bam
    # primary
    samtools view $file \
        | grep -P "^\S+\s16\s" \
        | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
        | cat /home/result/header -| samtools view -bS ->/home/result/primary_rev_STAR_align_${readfile}.bam
    samtools index /home/result/primary_rev_STAR_align_${readfile}.bam
done
