#!/bin/bash

# /Price_1.0.3b/gedi -e Price -h

genomefa=/home/ref/ensembl_109_hsa/Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa
annotgtf=/home/ref/ensembl_109_hsa/Homo_sapiens.GRCh38.109.gtf

gzip -d ${genomefa}.gz
gzip -d ${annotgtf}.gz

# prepare genome indexes
/Price_1.0.3b/gedi -e IndexGenome \
    -s $genomefa \
    -a $annotgtf \
    -f /home/result \
    -nobowtie \
    -nostar \
    -p

mv /home/saori/.gedi/genomic/Homo_sapiens.GRCh38.109.oml .

#######################
mappedFile1=/home/data/STAR_genome_align_13823_9601_184391_HVG2NBGXN_1_TGTTGACT_R1_Aligned.sortedByCoord.out.bam
mappedFile2=/home/data/STAR_genome_align_13823_9601_184392_HVG2NBGXN_2_ACGGAACT_R1_Aligned.sortedByCoord.out.bam
mappedFile3=/home/data/STAR_genome_align_13823_9601_184393_HVG2NBGXN_3_TCTGACAT_R1_Aligned.sortedByCoord.out.bam
mappedFile4=/home/data/STAR_genome_align_13823_9601_184394_HVG2NBGXN_4_CGGGACGG_R1_Aligned.sortedByCoord.out.bam

indexGediGenome=/home/result/Homo_sapiens.GRCh38.109.oml

mkdir price_all
# 1) Make a bamlist (order = column order)
cat > price_all/price.bamlist <<'EOF'
/home/data/STAR_genome_align_13823_9601_184391_HVG2NBGXN_1_TGTTGACT_R1_Aligned.sortedByCoord.out.bam
/home/data/STAR_genome_align_13823_9601_184392_HVG2NBGXN_2_ACGGAACT_R1_Aligned.sortedByCoord.out.bam
/home/data/STAR_genome_align_13823_9601_184393_HVG2NBGXN_3_TCTGACAT_R1_Aligned.sortedByCoord.out.bam
/home/data/STAR_genome_align_13823_9601_184394_HVG2NBGXN_4_CGGGACGG_R1_Aligned.sortedByCoord.out.bam
EOF

/Price_1.0.3b/gedi -e Price \
    -reads price_all/price.bamlist  \
    -genomic $indexGediGenome \
    -prefix price_all/price2 \
    -progress -plot -percond
