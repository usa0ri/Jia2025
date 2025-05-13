
from pathlib import Path

cur_dir = Path(__file__).parent
save_dir = cur_dir / "result_transRNA-seq"
if not save_dir.exists():
    save_dir.mkdir(parents=True)

from myRiboSeq import myprep_bin
import myRiboSeq.myUtil as my

ref = myprep_bin.prep_data(
    save_dir=save_dir,
    ref_dir = 'ref/Homo_sapiens_109_saori',
    data_dir="data",
    sp='hsa'
    )

smpls = ref.exp_metadata.df_metadata['sample_name'].values
cols,cols_hex = my._get_spaced_colors(len(smpls))
col_list = dict(zip(smpls,cols_hex))

##################
from myRiboSeq import myTransRNA

myTransRNA.region_mapped(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls=smpls
)

myTransRNA.norm_density3(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls=smpls
)

myTransRNA.indv_plot(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls=smpls,
    full_align=True
)
myTransRNA.indv_plot2(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    ref=ref,
    smpls=smpls,
    full_align=True,
    tr_list=[
        'ENST00000366923',
        'ENST00000272102',
        'ENST00000337288'
        ]
)
