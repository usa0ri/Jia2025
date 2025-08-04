
from pathlib import Path, PosixPath
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import re

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import myRiboSeq.myRiboBin as mybin
import myRiboSeq.myUtil as my
import myRiboSeq.myRef as myref


def rpkm(
    save_dir:PosixPath,
    load_dir:PosixPath,
    ref:myref.Ref,
    smpls:list,
    args=my.default_args
):
    save_dir = save_dir / 'rpkm'
    if not save_dir.exists():
        save_dir.mkdir()
    
    args = my._parse_args(args)

    for s in smpls:
        print(f'calculating RPKM in {s}...')
        outfile_name = save_dir / f'rpkm_{s}.csv.gz'
        obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1)

        dict_tr,_ = my._filter_reads(
            dict_tr=dict_tr,
            args=args
        )

        count_sum = len(df_data)

        rpkm = np.array([
            len(v) / (count_sum * ref.annot.annot_dict[v['tr_id'].iloc[0]]["cdna_len"]) * 1e+9
            for v in dict_tr.values()
        ])
        tpm_ = np.array([
            (len(v)/ref.annot.annot_dict[v['tr_id'].iloc[0]]["cdna_len"])
            for v in dict_tr.values()
        ])
        tpm = tpm_ / np.sum(tpm_,axis=None) * 1e+6

        pd.DataFrame({'rpkm':rpkm,'tpm':tpm},index=dict_tr.keys()).\
            to_csv(outfile_name,compression="gzip")



def _scatter_genes(
    rpkms,
    pair,
    threshold,
    c,
    pdf,
    texts = '',
    target_genes:list = []
):
    rpkms.loc[:,f'{c}_ratio'] = rpkms.loc[:,f'{c}_{pair[0]}'] / rpkms.loc[:,f'{c}_{pair[1]}']
    if threshold>0:
        rpkms.loc[:,f'{c}_label'] = rpkms.\
            apply(lambda x: 'high' if (x[f'{c}_ratio'] > threshold) \
                  else ('low' if x[f'{c}_ratio'] < 1/threshold else 'medium'), axis=1)
    else:
        rpkms.loc[:,f'{c}_label'] = 'medium'
    # rpkms.loc[:,f'{c}_label'] = rpkms.\
    #     apply(lambda x: x[f'{c}_label'] if (x[f'{c}_{pair[0]}'] > threshold_tr and x[f'{c}_{pair[1]}'] > threshold_tr) else 'medium', axis=1)
    
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.scatter(
        np.log2(rpkms.query(f'{c}_label == "medium"')[f'{c}_{pair[1]}']),
        np.log2(rpkms.query(f'{c}_label == "medium"')[f'{c}_{pair[0]}']),
        color="#3E3E3E",
        s=1)
    if threshold>0:
        ax.scatter(
            np.log2(rpkms.query(f'{c}_label == "high"')[f'{c}_{pair[1]}']),
            np.log2(rpkms.query(f'{c}_label == "high"')[f'{c}_{pair[0]}']),
            color="#BF0D0D",
            s=1)
        ax.scatter(
            np.log2(rpkms.query(f'{c}_label == "low"')[f'{c}_{pair[1]}']),
            np.log2(rpkms.query(f'{c}_label == "low"')[f'{c}_{pair[0]}']),
            color="#0D1BBF",
            s=1)
    if len(target_genes) > 0:
        for tr in target_genes:
            if tr in rpkms.index:
                ax.scatter(
                    np.log2(rpkms.loc[tr,f'{c}_{pair[1]}']),
                    np.log2(rpkms.loc[tr,f'{c}_{pair[0]}']),
                    color="#FF0000",
                    s=5
                )
                # ax.text(
                #     np.log2(rpkms.loc[tr,f'{c}_{pair[1]}']),
                #     np.log2(rpkms.loc[tr,f'{c}_{pair[0]}']),
                #     rpkms.loc[tr,'name'],
                #     fontsize=8,
                #     color="#FF0000"
                # )
    ax.set_xlabel(f'{pair[1]} (log2 {c})',fontsize=10)
    ax.set_ylabel(f'{pair[0]} (log2 {c})',fontsize=10)
    max_lim = np.amax([ax.get_xlim()[1],ax.get_ylim()[1]])
    min_lim = np.amin([ax.get_xlim()[0],ax.get_ylim()[0]])
    ax.set_xticks(list(range(0,int(round(max_lim,0))+1,5)))
    ax.set_yticks(list(range(0,int(round(max_lim,0))+1,5)))
    ax.set_xlim([0,max_lim])
    ax.set_ylim([0,max_lim])
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if texts != '':
        ax.text(1,max_lim-1,texts,fontsize=10)
    fig.tight_layout()

    fig.savefig(pdf,format='pdf')
    plt.close()

    return rpkms

def scatter_rpkm(
    save_dir:PosixPath,
    load_dir:PosixPath,
    pairs:list,
    ref:myref.Ref,
    threshold=0,
    fname = '',
    mode='rpkm',
    target_genes:list = []
):
    save_dir = save_dir / 'scatter_rpkm'
    if not save_dir.exists():
        save_dir.mkdir()
    # scatter plots
    if threshold>0:
        outfile_name = save_dir / f'scatter_{mode}{fname}_fc{threshold}.pdf'
    else:
        outfile_name = save_dir / f'scatter_{mode}{fname}.pdf'
    
    pdf = PdfPages(outfile_name)
    for pair in pairs:
        print(f'\ncalculating {mode} ratio {pair[0]} / {pair[1]}...')
        outfile_name = save_dir / f'rpkm_ratio_{pair[0]}_{pair[1]}'
        rpkm1 = pd.read_csv(load_dir / f'rpkm_{pair[0]}.csv.gz',header=0,index_col=0)
        rpkm2 = pd.read_csv(load_dir / f'rpkm_{pair[1]}.csv.gz',header=0,index_col=0)
        rpkms = pd.merge(rpkm1,rpkm2,
            left_index=True, right_index=True,
            how='inner',suffixes=[f'_{p}' for p in pair])
        r = pearsonr(
            rpkms[f'{mode}_{pair[0]}'],
            rpkms[f'{mode}_{pair[1]}'])
        rpkms['name'] = [
            ref.id.dict_name[tr]['symbol']
            for tr in rpkms.index
        ]
        rpkms = _scatter_genes(
            rpkms=rpkms,
            pair=pair,
            pdf=pdf,
            c=mode,
            threshold=threshold,
            texts='',
            target_genes=target_genes
        )
        # texts=f'Pearson\'s r = {round(r.correlation,2)}',
        # extract target genes
        if len(target_genes) > 0:
            rpkms.loc[target_genes,:].to_csv(
                save_dir / f'{mode}_{pair[0]}_{pair[1]}_target_genes.csv.gz',
                compression="gzip"
            )
        
        rpkms.to_csv(outfile_name.with_suffix('.csv.gz'),compression="gzip")
        # stat
        outfile_name = save_dir / f'{mode}_ratio_stat_{pair[0]}_{pair[1]}'
        rpkms_label = rpkms.query(f"{mode}_ratio>0").pivot_table(values=f"{mode}_ratio",index=f"{mode}_label",aggfunc=len)
        rpkms_label[f"{mode}_percent"] = rpkms_label[f"{mode}_ratio"] / rpkms_label[f"{mode}_ratio"].sum() *100
        rpkms_label.to_csv(outfile_name.with_suffix('.csv.gz'),compression="gzip")
    pdf.close()
        