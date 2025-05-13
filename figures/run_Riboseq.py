
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

myprep_bin.calc_norm_ribo_density(
    save_dir=save_dir,
    load_dir=save_dir / 'prep_data',
    is_length=False,
    smpls=smpls,
    is_norm=True,
    ref=ref
)


from myRiboSeq import myRibo

myRibo.aggregation_ribo(
    save_dir=save_dir / 'Longfei2023',  
    load_dir=save_dir / 'norm_ribo_density',
    plot_range5=(-30,285),
    plot_range3=(-300,15),
    col_list=col_list,
    labels=[
        'Control',
        'Starvation',
        'Starvation +\n $\mathit{trans}$-RNA'
        ],
    fname='aggregation_hek293'
)

myRibo.ATF4_plot(
    save_dir=save_dir / 'Longfei2023',
    load_dir=save_dir / 'norm_ribo_density',
    fname='ATF4_plot_hek293',
    col_list=col_list,
    labels=[
        'Control',
        'Starvation',
        'Starvation +\n $\mathit{trans}$-RNA'
        ],
    ref=ref
)

myRibo.ratio_5utr_cds(
    save_dir=save_dir / 'Longfei2023',
    load_dir=save_dir / 'prep_data',
    ref=ref,
    pairs=[
        ('Ribo_starvation_trans1','Ribo_starvation'),
        ('Ribo_starvation_trans1','Ribo_starvation_trans2'),
    ],
    labels=[
        ('Starvation + $\mathit{trans}$-RNA 1','Starvation'),
        ('Starvation + $\mathit{trans}$-RNA 1','Starvation + $\mathit{trans}$-RNA 2'),
    ]

)
