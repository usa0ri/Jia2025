
from pathlib import Path, PosixPath
import joblib
import pickle
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, pearsonr
import re
import gzip
import itertools
import pysam
from tqdm import tqdm
import statsmodels.api as sm
# import venn

from myRiboSeq import myRef as myref
from myRiboSeq import myUtil as my

color_region = {
    '5UTR':"#9FE2BF",
    'CDS':"#6495ED",
    '3UTR':"#CD5C5C",
    "denom":"#818589",
    "numer":"#C41E3A"
}

color_frame = {
    'frame0':"#F8766D",
    'frame1':"#00BA38",
    'frame2':"#619CFF"
}

# thresholds for total counts of transcripts
thresholds = [np.inf,64,32,16,8,0]
codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = font_prop.get_name()
    
def reads_price_orfs(
    save_dir:PosixPath,
    load_price_path:PosixPath,
    dict_bam:dict,
    suffix=''
    ):

    save_dir = save_dir / 'PRICE_res'
    if not save_dir.exists():
        save_dir.mkdir()
    
    df_orf_merge = pd.read_csv( load_price_path, sep='\t', index_col=0, header=0 )
    smpls = [c.split('Type_')[1] for c in df_orf_merge.columns if 'Type_' in c]
    
    df_counts = pd.DataFrame()
    dict_orf_lengths = {}
    for smpl, bam in dict_bam.items():

        infile = pysam.AlignmentFile( bam, 'rb' )

        dict_counts_smpl = {}
        i = 0
        for location in tqdm(df_orf_merge['Location'],desc=f'counting reads of {smpl}...'):
            
            chrom_strand = location.split(':')[0]
            chrom = chrom_strand[:-1]
            strand = chrom_strand[-1]
            starts = [int(x.split('-')[0]) for x in location.split(':')[1].split('|')]
            ends = [int(x.split('-')[1]) for x in location.split(':')[1].split('|')]

            cnts = 0;orf_len = 0
            for start, end in zip(starts, ends):
                # count reads in the specified region
                cnts += infile.count(
                    contig=chrom,
                    start=start,
                    end=end
                )
                orf_len += end - start
            
            dict_counts_smpl[i] = cnts
            dict_orf_lengths[i] = orf_len

            i += 1

        infile.close()

        df_counts = pd.merge(
            df_counts,
            pd.DataFrame().from_dict(dict_counts_smpl,orient='index',columns=[smpl]),
            how='outer',
            left_index=True,
            right_index=True
        )  
    df_counts['length'] = [
        dict_orf_lengths[i]
        for i in df_counts.index
    ]  
    df_counts = df_counts.set_axis(df_orf_merge['Id'],axis=0)
    df_counts.to_csv( save_dir / f'df_counts{suffix}.csv.gz', sep=',' )

def scatter_price_orfs(
    save_dir:PosixPath,
    load_price_count_path:PosixPath,
    load_price_orf_path:PosixPath,
    suffix=''
):
    df_counts = pd.read_csv( load_price_count_path, sep=',', index_col=0, header=0 )
    df_orf = pd.read_csv( load_price_orf_path, sep='\t', index_col=0, header=0 )
    df_orf.reset_index(inplace=True)
    df_orf.set_index('Id', inplace=True)

    idx_atf4 = [i for i,idx in enumerate(df_orf['Gene']) if 'ENSG00000128272' in idx]
    df_orf.iloc[idx_atf4,:].to_csv(save_dir / f'ATF4{suffix}.csv')

    # filter ORFs by padj
    df_orf['padj'] = sm.stats.multipletests(df_orf['p value'], method='fdr_bh')[1]
    df_orf.to_csv( save_dir / f'df_orf_price{suffix}.csv.gz', sep=',' )
    idx = df_orf['padj'] < 0.1
    ids = df_orf.index[idx].tolist()
    df_counts = df_counts.loc[ids,:]
    df_orf = df_orf.loc[ids,:]
    df_counts.to_csv(save_dir / f'df_counts_price{suffix}.csv.gz', sep=',')

    assert len(df_counts) == len(df_orf), \
        f'Number of ORFs ({len(df_orf)}) and counts ({len(df_counts)}) do not match.'

    # plot all the ATF4 ORFs
    idx_atf4 = [i for i,idx in enumerate(df_orf['Gene']) if 'ENSG00000128272' in idx]
    
    idx_atf4_uorf2 = [idx for idx in df_orf.index[idx_atf4] if 'uoORF' in idx][0]
    idx_atf4_morf = [idx for idx in df_orf.index[idx_atf4] if 'iORF' in idx][0]
    idx_atf4 = [idx_atf4_uorf2,idx_atf4_morf]

    ## plot
    pairs = [
        ['Control_Ribo','Starvation_Ribo'],
        ['Starvation_Ribo','Starvation_transATF4_Ribo1'],
        ['Starvation_Ribo','Starvation_transATF4_Ribo2'],
        ['Starvation_transATF4_Ribo1','Starvation_transATF4_Ribo2'],
    ]
    labels = [
        ['Control','Starvation'],
        ['Starvation','Starvation+transATF4_1'],
        ['Starvation','Starvation+transATF4_2'],
        ['Starvation+transATF4_1','Starvation+transATF4_2'],
    ]
    pdf = PdfPages(save_dir / f'scatter_price_orfs_Ribo{suffix}.pdf')
    ticks_now = [0,5,10]
    for pair,label in zip(pairs,labels):
        fig,ax = plt.subplots(1,1,figsize=(3,3))

        # fold change
        # fc = np.log2(1+df_orf[pair[1]] /  df_orf['length']) - np.log2(1+df_counts[pair[0]] / df_counts['length'])
        # pd.DataFrame(fc,index=df_counts.index,columns=['log2 fold change'])\
        #     .sort_values(by='log2 fold change',ascending=False)\
        #     .to_csv(save_dir / f'fc_{pair[0]}_{pair[1]}.csv.gz',sep=',')
        
        ax.scatter(
            np.log2(1+df_counts[pair[0]] /  df_counts['length']),
            np.log2(1+df_counts[pair[1]] / df_counts['length']),
            s=1,
            facecolors='black',
            edgecolors='black'
        )
        ax.plot(
            [ax.get_xlim()[0],ax.get_xlim()[1]],
            [ax.get_ylim()[0],ax.get_ylim()[1]],
            linestyle='--',
            color='#808080'
        )
        ax.scatter(
            np.log2(1+df_counts.loc[idx_atf4,pair[0]].values /  df_counts.loc[idx_atf4,'length'].values),
            np.log2(1+df_counts.loc[idx_atf4,pair[1]].values /  df_counts.loc[idx_atf4,'length'].values),
            s=5,
            facecolors='red',
            edgecolors='red',
        )
        # annotation
        idx = idx_atf4_uorf2
        ax.text(
            x=np.log2(1+df_counts.loc[idx, pair[0]] /  df_counts.loc[idx,'length']),
            y=np.log2(1+df_counts.loc[idx,pair[1]] /  df_counts.loc[idx,'length']),
            s='ATF4 uORF2',
            fontsize=8,
            color='red'
        )

        idx = idx_atf4_morf
        ax.text(
            x=np.log2(1+df_counts.loc[idx,pair[0]] /  df_counts.loc[idx,'length']),
            y=np.log2(1+df_counts.loc[idx,pair[1]] /  df_counts.loc[idx,'length']),
            s='ATF4 mORF',
            fontsize=8,
            color='red'
        )
        
        ax.set_xlabel(f'log2 (mean reads), {label[0]}')
        ax.set_ylabel(f'log2 (mean reads), {label[1]}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # same ticklabels between x and y
        ax.set_xticks(ticks_now)
        ax.set_xticklabels(labels=ticks_now,fontsize=10)
        ax.set_yticks(ticks_now)
        ax.set_yticklabels(labels=ticks_now,fontsize=10)

        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
    pdf.close()


def volcano_plot(
    save_dir:PosixPath,
    load_data:PosixPath,
    threshold_fc:float,
    threshold_pval:float,
    tr_highlight:dict,
    suffix=''
):
    save_dir = save_dir / 'volcano_plot'
    if not save_dir.exists():
        save_dir.mkdir()

    df = pd.read_csv( load_data, header=0, index_col=0 )

    p = 'pvalue'

    idx_high = df.query(f'(log2FoldChange >= {np.log2(threshold_fc)}) and ({p} <= {threshold_pval})').index
    idx_low = df.query(f'(log2FoldChange <= {np.log2(1/threshold_fc)}) and ({p} <= {threshold_pval})').index
    idx_medium = np.array([x for x in df.index if (x not in idx_high) and (x not in idx_low)])
    
    outfile_name = save_dir / f'volcano_plot{suffix}.pdf'
    pdf = PdfPages(outfile_name)

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    for g,c in zip([idx_medium,idx_high,idx_low],['#808080',"#FF0000","#0000FF"]):
        ax.scatter(
            df.loc[g,'log2FoldChange'],
            df.loc[g,p].apply(lambda x: -np.log10(x)),
            color=c,
            s=1
        )
    for tr,tr_name in tr_highlight.items():
        if tr in df.index:
            ax.scatter(
                df.loc[tr,'log2FoldChange'],
                -np.log10(df.loc[tr,p]),
                color="#F59F01",
                s=5
            )
            # ax.text(
            #     x=df.loc[tr,'log2FoldChange'],
            #         y=-np.log10(df.loc[tr,p]),
            #         s=tr_name,
            #         color='black',
            #         ha='left',va='bottom'
            #     )
    xlim_now = np.sqrt(np.max(np.array(ax.get_xlim())**2))
    ax.set_xlim(-xlim_now,xlim_now)
    ax.set_xlabel('Log2 fold change')
    ax.set_ylabel('-log10(p value)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig(pdf,format='pdf')
    pdf.close()

if __name__ == "__main__":

    save_dir = cur_dir.parent / "result/revision_transRNA"

    reads_price_orfs(
        save_dir=save_dir,
        load_price_path = save_dir / 'PRICE' / 'price_all' / 'price2.orfs.tsv.gz',
        dict_bam = {
            'Control_Ribo':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184391_HVG2NBGXN_1_TGTTGACT_R1_Aligned.sortedByCoord.out.bam',
            'Starvation_Ribo':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184392_HVG2NBGXN_2_ACGGAACT_R1_Aligned.sortedByCoord.out.bam',
            'Starvation_transATF4_Ribo1':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184393_HVG2NBGXN_3_TCTGACAT_R1_Aligned.sortedByCoord.out.bam',
            'Starvation_transATF4_Ribo2':save_dir / 'preprocessing' / 'STAR_genome_align_13823_9601_184394_HVG2NBGXN_4_CGGGACGG_R1_Aligned.sortedByCoord.out.bam'
        },
        suffix='_new2'
    )

    scatter_price_orfs(
        save_dir=save_dir,
        load_price_count_path = save_dir / 'PRICE_res' / 'df_counts_new.csv.gz',
        load_price_orf_path = save_dir / 'PRICE_res' / 'df_orf_merge.csv.gz',
    )

    volcano_plot(
    save_dir=save_dir / 'DEseq2',
    load_data = save_dir / 'DEseq2' / 'res.csv',
    threshold_fc = 1.5,
    threshold_pval = 0.1,
    tr_highlight = {
        'ENST00000680748_uoORF_0':'ATF4 uORF2',
        'ENST00000404241_uoORF_5':'ATF4 uORF2',
        'ENST00000680748_uoORF_7':'ATF4 uORF1',
        'ENST00000675582_Variant_3':'ATF4 uORF2',
        # 'ENST00000337304_iORF_10':'ATF4 mORF'
    },
    suffix='_new'
)
    pass