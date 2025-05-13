
from pathlib import Path

cur_dir = Path(__file__)
save_dir = cur_dir.parent / "result"
if not save_dir.exists():
    save_dir.mkdir(parents=True)

from myRiboSeq import myprep_bin

ref = myprep_bin.prep_data(
    save_dir=save_dir,
    ref_dir = 'ref/Homo_sapiens_saori',
    data_dir="data",
    sp='hsa')

##################
from myRiboSeq import myTransRNA

myTransRNA.detect_peaks_GTI(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls_CHX=('Ribo-seq_rep1','Ribo-seq_rep2'),
    smpls_LTM=('GTI-seq_rep1','GTI-seq_rep2')
)

myTransRNA.indv_plot_ribo2(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls=[smpls[-1]],
    tr_list=[
        'ENST00000272102',
        ]
)
print("hoge")