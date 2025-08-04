
from pathlib import Path

cur_dir = Path(__file__)
save_dir = cur_dir.parent / "result"
if not save_dir.exists():
    save_dir.mkdir(parents=True)

from myRiboSeq import myRef as myref
from myRiboSeq import myRibo

myRibo.get_price_results_(
    save_dir=save_dir,
    price_suffix_list=[
        Path(save_dir / 'PRICE' / 'price_Control_Ribo/price_Control_Ribo'),
        Path(save_dir / 'PRICE' / 'price_Starvation_Ribo/price_Starvation_Ribo'),
        Path(save_dir / 'PRICE' / 'price_Starvation_transATF4_Ribo1/price_Starvation_transATF4_Ribo1'),
        Path(save_dir / 'PRICE' / 'price_Starvation_transATF4_Ribo2/price_Starvation_transATF4_Ribo2'),
    ]
)

myRibo.reads_price_orfs(
    save_dir=save_dir,
    load_price_path = save_dir / 'PRICE_res' / 'df_orf_merge.csv.gz',
    dict_bam = {
        'Control_Ribo':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184391_HVG2NBGXN_1_TGTTGACT_R1_Aligned.sortedByCoord.out.bam',
        'Starvation_Ribo':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184392_HVG2NBGXN_2_ACGGAACT_R1_Aligned.sortedByCoord.out.bam',
        'Starvation_transATF4_Ribo1':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184393_HVG2NBGXN_3_TCTGACAT_R1_Aligned.sortedByCoord.out.bam',
        'Starvation_transATF4_Ribo2':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184394_HVG2NBGXN_4_CGGGACGG_R1_Aligned.sortedByCoord.out.bam'
    }
)

myRibo.scatter_price_orfs(
    save_dir=save_dir,
    load_price_count_path = save_dir / 'PRICE_res' / 'df_counts.csv.gz',
    load_price_orf_path = save_dir / 'PRICE_res' / 'df_orf_merge.csv.gz',
)

print("hoge")

#######################
# RNA-seq analysis

from myRiboSeq import myPrepData

ref = myPrepData.prep_data(
    save_dir=save_dir,
    ref_dir = 'ref/Mus_musculus_109_saori',
    data_dir="/home/data/NextSeq/data20221117_Longfei_Trans",
    sp='mmu',
    is_return_ref = True
    )

from myRiboSeq import myDEG

myDEG.rpkm(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    args={
        'is_filter':False
    },
    smpls=['Control_RNA', 'Fasting_RNA', 'Fasting_transATF4_RNA'],
)

myDEG.scatter_rpkm(
    save_dir=save_dir,
    load_dir=save_dir / 'rpkm',
    ref=ref,
    pairs=[
        ('Fasting_RNA','Control_RNA'),
        ('Fasting_transATF4_RNA','Control_RNA'),
    ],
    threshold=2,
    mode='tpm',
)