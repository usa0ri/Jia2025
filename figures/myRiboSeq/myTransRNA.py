import pandas as pd
from pathlib import Path
import numpy as np
import gzip
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pprint
import seaborn as sns
import logomaker as lm
import venn
from sklearn.cluster import AgglomerativeClustering
import matplotlib as mpl
from matplotlib_venn import venn2

import mylib_bin as my
import myRiboBin as mybin

# thresholds for total counts of transcripts
thresholds = [np.inf,64,32,16,8,0]
atcg = ['A','T','C','G']
dict_cigar = {
    0:'M',
    1:'I',
    2:'D',
    3:'N',
    4:'S',
    5:'H',
    6:'P',
    7:'=',
    8:'X',
    9:'B'
}

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


def read_length_distribution(
    save_dir:Path,
    load_dir:Path,
    ref:my.Ref,
    smpls:list
):
    save_dir = save_dir / 'read_length_distribution'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if ref.exp_metadata.df_metadata.index.name != 'sample_name':
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    df_lengths = []
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
        idx_full = df_data['length'] == df_data['read_length']
        df_data = df_data.iloc[ idx_full.values, : ]

        df_length = df_data.groupby(['read_length']).apply(len)
        df_length.name = s

        if len(df_lengths) == 0:
            df_lengths = df_length
        else:
            df_lengths = pd.merge(
                df_lengths,df_length,
                how='outer',left_index=True,right_index=True)
    df_lengths.fillna(value=0,inplace=True)
    df_lengths = df_lengths.astype(int)

    df_lengths.to_csv(save_dir / 'plot_data.csv')

    def _plot(
        outfile_name:Path,
        df_lengths:pd.DataFrame,
        ref:my.Ref
    ):
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(2,1,figsize=(3,5),sharex=True,sharey=True)
        # for ax in axs:
        #     ax.hlines(0,xmin=10,xmax=160,colors='#808080',linestyle='dashed')
        for s in df_lengths.columns:
            if ref.exp_metadata.df_metadata.loc[s,'pool'] == 'total':
                idx_axs = 0
                axs[idx_axs].set_title('total')
            else:
                idx_axs = 1
                axs[idx_axs].set_title('cap')
            if ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense':
                col_now = '#880000'
                sign_now = 1
            else:
                col_now = '#000088'
                sign_now = -1
            (df_lengths[s]*sign_now).plot(use_index=True,ax=axs[idx_axs],color=col_now)

        for ax in axs:
            ax.legend(labels=['sense','antisense'],loc='upper right')
            ax.set_xlabel('read length')
            ax.set_ylabel('reads')

        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
        pdf.close()


    if 'method' in ref.exp_metadata.df_metadata.columns:
        for m in ref.exp_metadata.df_metadata['method'].unique():
            for c in ref.exp_metadata.df_metadata['condition'].unique():
                outfile_name = save_dir / f'read_length_{m}_{c}.pdf'
                idx_plot = (ref.exp_metadata.df_metadata['condition'].values == c) *\
                    (ref.exp_metadata.df_metadata['method'].values == m)
                # assert np.sum(idx_plot) == 4
                _plot(
                    outfile_name=outfile_name,
                    df_lengths=df_lengths.iloc[:,idx_plot],
                    ref=ref
                )
    
    else:

        for c in ref.exp_metadata.df_metadata['condition'].unique():
            outfile_name = save_dir / f'read_length_{c}.pdf'
            idx_plot = (ref.exp_metadata.df_metadata['condition'].values == c)
            # assert np.sum(idx_plot) == 4
            _plot(
                    outfile_name=outfile_name,
                    df_lengths=df_lengths.iloc[:,idx_plot],
                    ref=ref
                )

def region_mapped(
    save_dir,
    load_dir,
    ref,
    smpls
):
    save_dir = save_dir / 'region_mapped'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if ref.exp_metadata.df_metadata.index.name != 'sample_name':
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    df_regions = {}
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_cds=True)
        idx_full = df_data['length'] == df_data['read_length']
        df_data = df_data.iloc[ idx_full.values, : ]

        df_region = df_data.groupby(['cds_label']).apply(len)
        df_region.name = ref.exp_metadata.df_metadata.loc[s,'strand']
        p = ref.exp_metadata.df_metadata.loc[s,'pool']
        c = ref.exp_metadata.df_metadata.loc[s,'condition']
        if 'method' in ref.exp_metadata.df_metadata.columns:
            m = ref.exp_metadata.df_metadata.loc[s,'method']
            if m in df_regions.keys():
                if c in df_regions[m].keys():
                    if p in df_regions[m][c].keys():
                        df_regions[m][c][p] = pd.merge(
                            df_regions[m][c][p],df_region,
                            how='inner',left_index=True,right_index=True,
                            suffixes=[f'_{df_regions[m][c][p].name}',f'_{df_region.name}']
                        )
                    else:
                        df_regions[m][c][p] = df_region
                else:
                    df_regions[m][c] = {}
                    df_regions[m][c][p] = df_region
            else:
                df_regions[m] = {}
                df_regions[m][c] = {}
                df_regions[m][c][p] = df_region
        
        else:
            if c in df_regions.keys():
                if p in df_regions[c].keys():
                    df_regions[c][p] = pd.merge(
                        df_regions[c][p],df_region,
                        how='inner',left_index=True,right_index=True,
                        suffixes=[f'_{df_regions[c][p].name}',f'_{df_region.name}']
                    )
                else:
                    df_regions[c][p] = df_region
            else:
                df_regions[c] = {}
                df_regions[c][p] = df_region

    def func(pct,allvals):
        absolute = int(np.round(pct/100*np.sum(allvals)))
        if pct>1:
            out = f'{pct:.1f}% ({absolute:d})'
        else:
            out = ''
        return out
    
    def _plot(
        outfile_name:Path,
        v:dict
    ):
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(1,3,figsize=(8,3))
        
        idx_plotted = [False,False]
        for kk,vv in v.items():
            if kk == 'total':
                idx_axs = 0
                axs[idx_axs].set_title('total',pad=50)
                idx_plotted[idx_axs] = True
            else:
                idx_axs = 1
                axs[idx_axs].set_title('cap',pad=50)
                idx_plotted[idx_axs] = True

            vals = [
                    vv.loc['CDS','sense'],
                    vv.loc['5UTR','sense'],
                    vv.loc['3UTR','sense'],
                    vv.loc['CDS','antisense'],
                    vv.loc['5UTR','antisense'],
                    vv.loc['3UTR','antisense'],
                ]
            val_labels = [
                'CDS (sense)',
                '5UTR (sense)',
                '3UTR (sense)',
                'CDS (antisense)',
                '5UTR (antisense)',
                '3UTR (antisense)',
            ]
            wedges,texts,autotexts = axs[idx_axs].pie(
                x=vals,
                colors=[
                    "#6495ED",
                    "#9FE2BF",
                    "#CD5C5C",
                    "#4D6EA9",
                    "#73A48A",
                    "#9C4646",
                ],
                autopct=lambda pct: func(pct,vals),
                startangle=90,
                radius=1.8*1.2
            )
            plt.setp(
                autotexts,
                size=8
            )
        axs[2].set_axis_off()
        axs[2].legend(
            wedges,val_labels,
            bbox_to_anchor = (0.5,0.,0.5,1),
            bbox_transform = fig.transFigure,
            loc='center right')
        for i,idx in enumerate(idx_plotted):
            if not idx:
                axs[i].set_axis_off()
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
        pdf.close()
        

    if 'method' in ref.exp_metadata.df_metadata.columns:

        for k_,v_ in df_regions.items():
            for k,v in v_.items():
                outfile_name = save_dir / f'region_mapped_{k_}_{k}.pdf'
                _plot(outfile_name=outfile_name,v=v)
    
    else:
    
        for k,v in df_regions.items():
            outfile_name = save_dir / f'region_mapped_{k}.pdf'
            _plot(outfile_name=outfile_name,v=v)
            

def norm_density(
    save_dir,
    load_dir,
    ref,
    smpls
):
    save_dir = save_dir / 'norm_density'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if ref.exp_metadata.df_metadata.index.name != 'sample_name':
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    dict_dens_cds = {};dict_dens_utr5 = {};dict_dens_utr3 = {}
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_cds=True)
        idx_full = df_data['length'] == df_data['read_length']
        df_data = df_data.iloc[ idx_full.values, : ]
        dict_tr_now = list(df_data.groupby('tr_id'))

        dens_cds = np.zeros(101).astype(int)
        dens_utr5 = np.zeros(51).astype(int)
        dens_utr3 = np.zeros(51).astype(int)

        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{len(dict_tr_now)} transcript...',end='')
            len_utr5 = ref.annot.annot_dict[tr]['start']
            len_cds = ref.annot.annot_dict[tr]['cds_len']
            len_utr3 = ref.annot.annot_dict[tr]['cdna_len'] - len_utr5 - len_cds

            pos_cds = df['dist5_start'].values / len_cds
            pos_cds = pos_cds[ (pos_cds>=0)*(pos_cds<=1) ]
            dens_cds[ (pos_cds*100).astype(int) ] += 1

            if len_utr5 != 0:
                pos_utr5 = (df['dist5_start'].values + len_utr5) / len_utr5
                pos_utr5 = pos_utr5[ (pos_utr5>=0)*(pos_utr5<=1) ]
                dens_utr5[ (pos_utr5*50).astype(int) ] += 1

            if len_utr3 != 0:
                pos_utr3 = df['dist5_stop'].values / len_utr3
                pos_utr3 = pos_utr3[ (pos_utr3>=0)*(pos_utr3<=1) ]
                dens_utr3[ (pos_utr3*50).astype(int) ] += 1
        
        dict_dens_cds[s] = dens_cds
        dict_dens_utr5[s] = dens_utr5
        dict_dens_utr3[s] = dens_utr3
    
    for c in ref.exp_metadata.df_metadata['condition'].unique():
        outfile_name = save_dir / f'{c}.pdf'
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(2,3,figsize=(6,4))
        # for ax in axs:
        #     ax.hlines(0,xmin=10,xmax=160,colors='#808080',linestyle='dashed')
        for s in dict_dens_cds.keys():
            if ref.exp_metadata.df_metadata.loc[s,'condition'] != c:
                continue
            if ref.exp_metadata.df_metadata.loc[s,'pool'] == 'total':
                idx_axs = 0
            else:
                idx_axs = 1

            if ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense':
                col_now = '#880000'
                sign_now = 1
            else:
                col_now = '#000088'
                sign_now = -1

            axs[idx_axs,1].plot(
                np.arange(start=0,stop=1.01,step=0.01),
                dict_dens_cds[s]*sign_now,
                color=col_now
            )
            axs[idx_axs,0].plot(
                np.arange(start=0,stop=1.01,step=0.02),
                dict_dens_utr5[s]*sign_now,
                color=col_now
            )
            axs[idx_axs,2].plot(
                np.arange(start=0,stop=1.01,step=0.02),
                dict_dens_utr3[s]*sign_now,
                color=col_now
            )
        for i,title_now in enumerate(['total','cap']):
            axs[i,0].set_title(f'{title_now} 5\'UTR')
            axs[i,1].set_title(f'{title_now} CDS')
            axs[i,2].set_title(f'{title_now} 3\'UTR')
        
        for i in range(3):
            ylim_max = np.max([axs[0,i].get_ylim()[1], axs[1,i].get_ylim()[1]])
            ylim_min = np.min([axs[0,i].get_ylim()[0], axs[1,i].get_ylim()[0]])
            axs[0,i].set_ylim(ylim_min,ylim_max)
            axs[1,i].set_ylim(ylim_min,ylim_max)
        
        axs[-1,0].set_xlabel('position')
        axs[-1,0].set_ylabel('reads')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.suptitle(c)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
        pdf.close()

def norm_density2(
    save_dir,
    load_dir,
    ref,
    smpls
):
    save_dir = save_dir / 'norm_density2'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if ref.exp_metadata.df_metadata.index.name != 'sample_name':
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    dict_dens_cds = {};dict_dens_utr5 = {};dict_dens_utr3 = {}
    for s in smpls:
        if ref.exp_metadata.df_metadata.loc[s,'condition'] != 'control':
            continue
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_cds=True)
        idx_full = df_data['length'] == df_data['read_length']
        df_data = df_data.iloc[ idx_full.values, : ]
        dict_tr_now = list(df_data.groupby('tr_id'))

        dens_cds = np.zeros(102)
        dens_utr5 = np.zeros(100)
        dens_utr3 = np.zeros(100)

        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{len(dict_tr_now)} transcript...',end='')
            len_utr5 = ref.annot.annot_dict[tr]['start']
            len_cds = ref.annot.annot_dict[tr]['cds_len']
            len_utr3 = ref.annot.annot_dict[tr]['cdna_len'] - len_utr5 - len_cds

            pos_cds = df['dist5_start'].values / (len_cds-1) - 0.01/2
            pos_cds = pos_cds[ (pos_cds>=-0.015)*(pos_cds<1.005) ]
            dens_cds[ np.round(pos_cds*100).astype(int)+1 ] += 1

            if len_utr5 != 0:
                pos_utr5 = (df['dist5_start'].values + len_utr5) / len_utr5 - 0.01/2
                pos_utr5 = pos_utr5[ (pos_utr5>=-0.005)*(pos_utr5<0.995) ]
                dens_utr5[ np.round(pos_utr5*100).astype(int) ] += 1

            if len_utr3 != 0:
                pos_utr3 = df['dist5_stop'].values / (len_utr3-1) - 0.01/2
                pos_utr3 = pos_utr3[ (pos_utr3>=-0.005)*(pos_utr3<0.995) ]
                dens_utr3[ np.round(pos_utr3*100).astype(int) ] += 1
        
        dict_dens_cds[s] = dens_cds
        dict_dens_utr5[s] = dens_utr5
        dict_dens_utr3[s] = dens_utr3
    
    for c in ref.exp_metadata.df_metadata['condition'].unique():
        if c != 'control':
            continue

        outfile_name = save_dir / f'{c}.pdf'
        pdf = PdfPages(outfile_name)

        for smpl_ymax in ['all','CDS']:
            fig,axs = plt.subplots(2,1,figsize=(6,4))
            # for ax in axs:
            #     ax.hlines(0,xmin=10,xmax=160,colors='#808080',linestyle='dashed')
            for s in dict_dens_cds.keys():
                if ref.exp_metadata.df_metadata.loc[s,'condition'] != c:
                    continue
                if ref.exp_metadata.df_metadata.loc[s,'pool'] == 'total':
                    col_now = '#0B6F0A'
                else:
                    col_now = '#993007'

                if ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense':
                    sign_now = 1
                    idx_axs = 0
                else:
                    sign_now = -1
                    idx_axs = 1

                if smpl_ymax == 'all':
                    # adjust values at 5'UTR-CDS and CDS-3'UTR boundaries
                    dict_dens_cds_new = dict_dens_cds[s] * (dict_dens_utr5[s][-1]/dict_dens_cds[s][0])
                    dict_dens_utr3_new = dict_dens_utr3[s] * (dict_dens_cds_new[-1]/dict_dens_utr3[s][0])
                    # dict_dens_cds_new = dict_dens_cds[s]
                    # dict_dens_utr3_new = dict_dens_utr3[s]


                    axs[idx_axs].plot(
                        np.arange(start=-0.01,stop=1.01,step=0.01),
                        dict_dens_cds_new*sign_now,
                        color=col_now
                    )
                    axs[idx_axs].plot(
                        np.arange(start=-1,stop=0,step=0.01),
                        dict_dens_utr5[s]*sign_now,
                        color=col_now
                    )
                    axs[idx_axs].plot(
                        np.arange(start=1,stop=2.0,step=0.01),
                        dict_dens_utr3_new*sign_now,
                        color=col_now
                    )

                if smpl_ymax == 'CDS':
                    # adjust values at 5'UTR-CDS and CDS-3'UTR boundaries
                    dict_dens_cds_new = dict_dens_cds[s] * (dict_dens_utr5[s][-1]/dict_dens_cds[s][0])
                    dict_dens_utr3_new = dict_dens_utr3[s] * (dict_dens_cds_new[-1]/dict_dens_utr3[s][0])
                    # dict_dens_cds_new = dict_dens_cds[s]
                    # dict_dens_utr3_new = dict_dens_utr3[s]


                    axs[idx_axs].plot(
                        np.arange(start=-0.01,stop=1.01,step=0.01),
                        dict_dens_cds_new*sign_now,
                        color=col_now
                    )
                    axs[idx_axs].plot(
                        np.arange(start=-1,stop=0,step=0.01),
                        dict_dens_utr5[s]*sign_now,
                        color=col_now
                    )
                    axs[idx_axs].plot(
                        np.arange(start=1,stop=2.0,step=0.01),
                        dict_dens_utr3_new*sign_now,
                        color=col_now
                    )
                    
                    if sign_now<0:
                        axs[idx_axs].set_ylim([np.max(dict_dens_cds_new)*1.25*sign_now,axs[idx_axs].get_ylim()[1]])
                    else:
                        axs[idx_axs].set_ylim([axs[idx_axs].get_ylim()[0],np.max(dict_dens_cds_new)*1.25*sign_now])
        
            fig.suptitle(c)
            fig.tight_layout()
            fig.savefig(pdf,format='pdf')
            plt.close('all')
        pdf.close()


def norm_density3(
    save_dir,
    load_dir,
    ref,
    smpls
):
    save_dir = save_dir / 'norm_density3'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if ref.exp_metadata.df_metadata.index.name != 'sample_name':
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    dict_dens_cds = {};dict_dens_utr5 = {};dict_dens_utr3 = {}
    total_reads = {}
    for s in smpls:
        
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_cds=True)
        idx_full = df_data['length'] == df_data['read_length']
        df_data = df_data.iloc[ idx_full.values, : ]
        dict_tr_now = list(df_data.groupby('tr_id'))

        dens_cds = np.zeros(102)
        dens_utr5 = np.zeros(100)
        dens_utr3 = np.zeros(100)
        
        n_reads = 0
        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{len(dict_tr_now)} transcript...',end='')
            len_utr5 = ref.annot.annot_dict[tr]['start']
            len_cds = ref.annot.annot_dict[tr]['cds_len']
            len_utr3 = ref.annot.annot_dict[tr]['cdna_len'] - len_utr5 - len_cds

            pos_cds = df['dist5_start'].values / (len_cds-1) - 0.01/2
            pos_cds = pos_cds[ (pos_cds>=-0.015)*(pos_cds<1.005) ]
            dens_cds[ np.round(pos_cds*100).astype(int)+1 ] += 1
            n_reads += len(pos_cds)

            if len_utr5 != 0:
                pos_utr5 = (df['dist5_start'].values + len_utr5) / len_utr5 - 0.01/2
                pos_utr5 = pos_utr5[ (pos_utr5>=-0.005)*(pos_utr5<0.995) ]
                dens_utr5[ np.round(pos_utr5*100).astype(int) ] += 1
                n_reads += len(pos_utr5)

            if len_utr3 != 0:
                pos_utr3 = df['dist5_stop'].values / (len_utr3-1) - 0.01/2
                pos_utr3 = pos_utr3[ (pos_utr3>=-0.005)*(pos_utr3<0.995) ]
                dens_utr3[ np.round(pos_utr3*100).astype(int) ] += 1
                n_reads += len(pos_utr3)
        
        if 'method' in ref.exp_metadata.df_metadata.columns:
            m = ref.exp_metadata.df_metadata.loc[s,'method']
            if m not in dict_dens_cds.keys():
                dict_dens_cds[m] = {}
                dict_dens_utr5[m] = {}
                dict_dens_utr3[m] = {}
                total_reads[m] = {}
            dict_dens_cds[m][s] = dens_cds
            dict_dens_utr5[m][s] = dens_utr5
            dict_dens_utr3[m][s] = dens_utr3
            total_reads[m][s] = n_reads
        
        else:
            dict_dens_cds[s] = dens_cds
            dict_dens_utr5[s] = dens_utr5
            dict_dens_utr3[s] = dens_utr3
            total_reads[s] = n_reads
    
    def _plot(
            pdf,
            dict_dens_cds,dict_dens_utr5,dict_dens_utr3,total_reads,
            df_metadata):
        for is_norm in [False,True]:
            fig,axs = plt.subplots(2,1,figsize=(6,4))
            # for ax in axs:
            #     ax.hlines(0,xmin=10,xmax=160,colors='#808080',linestyle='dashed')
            for s in dict_dens_cds.keys():
                if df_metadata.loc[s,'condition'] != c:
                    continue
                if df_metadata.loc[s,'pool'] == 'total':
                    col_now = '#0B6F0A'
                else:
                    col_now = '#993007'

                if df_metadata.loc[s,'strand'] == 'sense':
                    sign_now = 1
                    idx_axs = 0
                else:
                    sign_now = -1
                    idx_axs = 1

                if is_norm:
                    axs[idx_axs].plot(
                        np.arange(start=-1,stop=2.0,step=0.01),
                        np.hstack((
                            dict_dens_utr5[s]*sign_now/total_reads[s]*1e+6,
                            dict_dens_cds[s][1:-1]*sign_now/total_reads[s]*1e+6,
                            dict_dens_utr3[s]*sign_now/total_reads[s]*1e+6)
                            ),
                        color=col_now,
                        label=df_metadata.loc[s,'pool']
                    )
                    fig.suptitle(f'{c} (with normalization)')
                    axs[idx_axs].set_ylabel('RPM density')
                
                else:
                    axs[idx_axs].plot(
                        np.arange(start=-1,stop=2.0,step=0.01),
                        np.hstack((
                            dict_dens_utr5[s]*sign_now,
                            dict_dens_cds[s][1:-1]*sign_now,
                            dict_dens_utr3[s]*sign_now)
                            ),
                        color=col_now,
                        label=df_metadata.loc[s,'pool']
                    )
                    fig.suptitle(f'{c} (without normalization)')
                    axs[idx_axs].set_ylabel('Read density')
                axs[idx_axs].set_xticks([-1,0,1,2])
                axs[idx_axs].set_xticklabels('')
                axs[idx_axs].set_xticks([-0.5,0.5,1.5],minor=True)
                axs[idx_axs].tick_params(axis='x',which='minor',length=0)
                axs[idx_axs].set_xticklabels(['5\'UTR','CDS','3\'UTR'],ha='center',minor=True)

            axs[0].legend(loc='upper right')
   
            fig.tight_layout()
            fig.savefig(pdf,format='pdf')
            plt.close('all')
    
    if 'method' in ref.exp_metadata.df_metadata.columns:
        for m in ref.exp_metadata.df_metadata['method'].unique():
            for c in ref.exp_metadata.df_metadata['condition'].unique():
                outfile_name = save_dir / f'norm_density_{m}_{c}.pdf'
                pdf = PdfPages(outfile_name)
                _plot(pdf,dict_dens_cds[m],dict_dens_utr5[m],dict_dens_utr3[m],total_reads[m],ref.exp_metadata.df_metadata)
                pdf.close()
    else:
        for c in ref.exp_metadata.df_metadata['condition'].unique():
            outfile_name = save_dir / f'norm_density_{c}.pdf'
            pdf = PdfPages(outfile_name)
            _plot(pdf,dict_dens_cds,dict_dens_utr5,dict_dens_utr3,total_reads,ref.exp_metadata.df_metadata)
            pdf.close()


def indv_plot(
    save_dir,
    load_dir,
    ref,
    smpls,
    fname='',
    full_align=True
):
    save_dir = save_dir / 'indv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    for c in ref.exp_metadata.df_metadata['condition'].unique():
        
        reads_trs = {};pos_trs = {}
        if not full_align:
            reads_nonalign_trs = {}
        for s in smpls:
            if ref.exp_metadata.df_metadata.loc[s,'condition'] != c:
                continue

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            if full_align:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
                idx_full = df_data['length'] == df_data['read_length']
                df_data = df_data.iloc[ idx_full.values, : ]
                dict_tr_now = list(df_data.groupby('tr_id'))
            else:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                dict_tr_now = list(df_data.groupby('tr_id'))

            tr_len = len(dict_tr_now)

            reads_tr = {};pos_tr = {}
            if not full_align:
                reads_nonalign_tr = {}
            for i,(tr,df) in enumerate(dict_tr_now):
                print(f'\r{i}/{tr_len} transcripts...',end='')

                reads = np.vstack(
                    [
                        df['dist5_start'].values,
                        df['dist5_start'].values + df['length'].values,
                    ]
                )
                    
                pos = np.arange(
                    start=-ref.annot.annot_dict[tr]['start'],
                    stop=ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['start']
                )
                reads_tr[tr] = reads
                pos_tr[tr] = pos

                if not full_align:
                    softclip5 = df.apply(
                        lambda row:row['read_seq'].find(row['seq']),
                        axis=1).values
                    reads_nonalign = np.vstack([
                        df['dist5_start'].values - softclip5,
                        df['dist5_start'].values,
                        df['dist5_start'].values + df['length'].values,
                        df['dist5_start'].values + df['read_length'].values - softclip5,
                    ])
                    reads_nonalign_tr[tr] = reads_nonalign
            
            reads_trs[s] = reads_tr
            pos_trs[s] = pos_tr
            if not full_align:
                reads_nonalign_trs[s] = reads_nonalign_tr
        
        smpls_now = list(reads_trs.keys())
        # FIXME
        if len(smpls_now) == 4:
            tr_list_all = list(set(list(reads_trs[smpls_now[0]].keys())) &\
                set(list(reads_trs[smpls_now[1]].keys())) &\
                set(list(reads_trs[smpls_now[2]].keys())) &\
                set(list(reads_trs[smpls_now[3]].keys())))
            tr_list_full = list(set(list(reads_trs[smpls_now[0]].keys())) |\
                set(list(reads_trs[smpls_now[1]].keys())) |\
                set(list(reads_trs[smpls_now[2]].keys())) |\
                set(list(reads_trs[smpls_now[3]].keys())))
        elif len(smpls_now) == 2:
            tr_list_all = list(set(list(reads_trs[smpls_now[0]].keys())) &\
                set(list(reads_trs[smpls_now[1]].keys())))
            tr_list_full = list(set(list(reads_trs[smpls_now[0]].keys())) |\
                set(list(reads_trs[smpls_now[1]].keys())))

        tr_list = list(set(tr_list_full) - set(tr_list_all))
        
        tr_list_names = [
            ref.id.dict_name[tr]['symbol']
            for tr in tr_list
        ]
        idx_sort = np.argsort(tr_list_names)
        tr_list = np.array(tr_list)[idx_sort]
        tr_list_names = np.array(tr_list_names)[idx_sort]

        if full_align:
            outfile_name = save_dir / f'{c}{fname}.pdf'
        else:
            outfile_name = save_dir / f'all_{c}{fname}.pdf'
        
        tr_len = len(tr_list)
        pdf = PdfPages(outfile_name)
        for i,tr in enumerate(tr_list):
            print(f'\r{i}/{tr_len} transcripts...',end='')
            fig,axs = plt.subplots(2,1,figsize=(6,4))
            for s in smpls_now:
                if ref.exp_metadata.df_metadata.loc[s,'pool'] == 'total':
                    idx_axs = 0
                else:
                    idx_axs = 1
                if ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense':
                    col_now = '#880000'
                    sign_now = 1
                else:
                    col_now = '#000088'
                    sign_now = -1

                cnt_tmp = np.ones(ref.annot.annot_dict[tr]['cdna_len']).astype(int)
                axs[idx_axs].axvspan(
                    0,ref.annot.annot_dict[tr]['cds_len'],
                    color="#FFF8DC",alpha=0.5
                )
                axs[idx_axs].hlines(
                    0,-ref.annot.annot_dict[tr]['start'],
                    ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'],
                    color="#808080"
                )
                axs[idx_axs].set_title(
                    ref.exp_metadata.df_metadata.loc[s,'pool']
                )
                if tr in reads_trs[s]:
                    for j in range(reads_trs[s][tr].shape[1]):
                        pos1 = reads_trs[s][tr][0,j]
                        pos2 = reads_trs[s][tr][1,j]
                        pos1_ = pos1 + ref.annot.annot_dict[tr]['start']
                        pos2_ = pos2 + ref.annot.annot_dict[tr]['start']
                        axs[idx_axs].plot(
                            np.arange(pos1,pos2),
                            [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2_-pos1_),
                            color=col_now
                        )
                        
                        if not full_align:
                            pos1n = reads_nonalign_trs[s][tr][0,j]
                            pos2n = reads_nonalign_trs[s][tr][1,j]
                            if pos1n != pos2n:
                                pos1n_ = pos1n + ref.annot.annot_dict[tr]['start']
                                axs[idx_axs].plot(
                                    np.arange(pos1n,pos2n),
                                    [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2n-pos1n),
                                    color="#808080"
                                )
                            pos1n = reads_nonalign_trs[s][tr][2,j]
                            pos2n = reads_nonalign_trs[s][tr][3,j]
                            if pos1n != pos2n:
                                axs[idx_axs].plot(
                                    np.arange(pos1n,pos2n),
                                    [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2n-pos1n),
                                    color="#808080"
                                )
                        cnt_tmp[ pos1_:pos2_ ] += 1
                
                ymax = np.amax(cnt_tmp)
                if ymax<=20:
                    axs[idx_axs].set_ylim([-20-1,20+1])
                    ymax = 20
                    yticks = list(range(-ymax,ymax+1,10))
                    ylabels = [y if y>0 else -y for y in yticks]
                else:
                    axs[idx_axs].set_ylim([-ymax-1,ymax+1])
                    ymax = int(np.ceil((ymax/10)*10))
                    yticks = [-ymax,0,ymax]
                    ylabels = [y if y>0 else -y for y in yticks]
                
                axs[idx_axs].set_yticks(yticks)
                axs[idx_axs].set_yticklabels(ylabels)
                axs[idx_axs].set_ylabel('reads')

            tr_name = ref.id.dict_name[tr]['symbol']
            fig.suptitle(f'{tr} {tr_name}')
            fig.tight_layout()
            fig.savefig(pdf,format='pdf')
            plt.close('all')
        pdf.close()

        if full_align:
            outfile_name = save_dir / f'full_transcript_list_{c}{fname}.csv'
        else:
            outfile_name = save_dir / f'transcript_list_{c}_{fname}.csv'
        pd.DataFrame({
            'number':np.arange(1,len(tr_list)+1),
            'transcript name':tr_list_names,
            f'read count in {c} total (sense)':[
                (reads_trs[smpls_now[0]][tr]).shape[1]
                if tr in reads_trs[smpls_now[0]] else 0
                for tr in tr_list],
            f'read count in {c} total (antisense)':[
                (reads_trs[smpls_now[2]][tr]).shape[1]
                if tr in reads_trs[smpls_now[2]] else 0
                for tr in tr_list],
            f'read count in {c} cap (sense)':[
                (reads_trs[smpls_now[1]][tr]).shape[1]
                if tr in reads_trs[smpls_now[1]] else 0
                for tr in tr_list],
            f'read count in {c} cap (antisense)':[
                (reads_trs[smpls_now[3]][tr]).shape[1]
                if tr in reads_trs[smpls_now[3]] else 0
                for tr in tr_list],
            
        },index=tr_list).to_csv(outfile_name)

        print("hoge")


def indv_plot_multi(
    save_dir,
    load_dir,
    ref,
    smpls,
    fname='',
    full_align=True
):
    save_dir = save_dir / 'indv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    reads_trs = {};pos_trs = {}
    if not full_align:
        reads_nonalign_trs = {}

    tr_list = [];cond_list = []
    for s in smpls:
        cond = ref.exp_metadata.df_metadata.loc[s,'condition']
        strand = ref.exp_metadata.df_metadata.loc[s,'strand']
        pool = ref.exp_metadata.df_metadata.loc[s,'pool']

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        if full_align:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
            idx_full = df_data['length'] == df_data['read_length']
            df_data = df_data.iloc[ idx_full.values, : ]
            dict_tr_now = list(df_data.groupby('tr_id'))
        else:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
            dict_tr_now = list(df_data.groupby('tr_id'))

        tr_len = len(dict_tr_now)

        reads_tr = {};pos_tr = {}
        if not full_align:
            reads_nonalign_tr = {}
        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            if len(df) < 8:
                continue

            reads = np.vstack(
                [
                    df['dist5_start'].values,
                    df['dist5_start'].values + df['length'].values,
                ]
            )
                    
            pos = np.arange(
                start=-ref.annot.annot_dict[tr]['start'],
                stop=ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['start']
            )
            reads_tr[tr] = reads
            pos_tr[tr] = pos

            if not full_align:
                softclip5 = df.apply(
                    lambda row:row['read_seq'].find(row['seq']),
                    axis=1).values
                reads_nonalign = np.vstack([
                    df['dist5_start'].values - softclip5,
                    df['dist5_start'].values,
                    df['dist5_start'].values + df['length'].values,
                    df['dist5_start'].values + df['read_length'].values - softclip5,
                ])
                reads_nonalign_tr[tr] = reads_nonalign
            
        reads_trs[(cond,pool,strand)] = reads_tr
        pos_trs[(cond,pool,strand)] = pos_tr
        if not full_align:
            reads_nonalign_trs[(cond,pool,strand)] = reads_nonalign_tr
        
        tr_list = list(set(tr_list) | set(list(reads_tr.keys())))
        if cond not in cond_list:
            cond_list.append(cond)
    
    tr_list_names = [
        ref.id.dict_name[tr]['symbol']
        for tr in tr_list
    ]
    idx_sort = np.argsort(tr_list_names)
    tr_list = np.array(tr_list)[idx_sort]
    tr_list_names = np.array(tr_list_names)[idx_sort]
    dict_cond_col = dict(zip(cond_list,list(range(len(cond_list)))))

    if full_align:
        outfile_name = save_dir / f'indv_plot_pileup{fname}.pdf'
    else:
        outfile_name = save_dir / f'indv_plot_pileup_partial{fname}.pdf'

    # output the count data
    dict_out = {
        'number':np.arange(1,len(tr_list)+1),
        'transcript name':tr_list_names
    }
    for cond in cond_list:
        for pool in ['total','cap']:
            for strand in ['sense','antisense']:
                dict_out[ f'read count in {cond} {pool} {strand}' ] = [
                    (reads_trs[(cond,pool,strand)][tr]).shape[1]
                    if tr in reads_trs[(cond,pool,strand)] else 0
                    for tr in tr_list
                ]
    pd.DataFrame(dict_out,index=tr_list).to_csv(outfile_name.with_suffix('.csv.gz'))

    # plot    
    tr_len = len(tr_list)
    pdf = PdfPages(outfile_name)
    for i,tr in enumerate(tr_list):
        print(f'\r{i}/{tr_len} transcripts...',end='')
        fig,axs = plt.subplots(2,len(cond_list),figsize=(6*len(cond_list),4),sharex=True,sharey=True)
        ymax_tr = 0
        for cond,pool,strand in reads_trs.keys():
            s = (cond,pool,strand)
            if pool == 'total':
                idx_axs = 0
            else:
                idx_axs = 1
            if strand == 'sense':
                col_now = '#880000'
                sign_now = 1
            else:
                col_now = '#000088'
                sign_now = -1
            idx_col = dict_cond_col[cond]

            cnt_tmp = np.ones(ref.annot.annot_dict[tr]['cdna_len']).astype(int)
            axs[idx_axs,idx_col].axvspan(
                0,ref.annot.annot_dict[tr]['cds_len'],
                color="#FFF8DC",alpha=0.5
            )
            axs[idx_axs,idx_col].hlines(
                0,-ref.annot.annot_dict[tr]['start'],
                ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'],
                color="#808080"
            )
            axs[idx_axs,idx_col].set_title(f'{pool}, {cond}')
            if tr in reads_trs[s]:
                for j in range(reads_trs[s][tr].shape[1]):
                    pos1 = reads_trs[s][tr][0,j]
                    pos2 = reads_trs[s][tr][1,j]
                    pos1_ = pos1 + ref.annot.annot_dict[tr]['start']
                    pos2_ = pos2 + ref.annot.annot_dict[tr]['start']
                    axs[idx_axs,idx_col].plot(
                        np.arange(pos1,pos2),
                        [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2_-pos1_),
                        color=col_now
                    )
                    
                    if not full_align:
                        pos1n = reads_nonalign_trs[s][tr][0,j]
                        pos2n = reads_nonalign_trs[s][tr][1,j]
                        if pos1n != pos2n:
                            pos1n_ = pos1n + ref.annot.annot_dict[tr]['start']
                            axs[idx_axs,idx_col].plot(
                                np.arange(pos1n,pos2n),
                                [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2n-pos1n),
                                color="#808080"
                            )
                        pos1n = reads_nonalign_trs[s][tr][2,j]
                        pos2n = reads_nonalign_trs[s][tr][3,j]
                        if pos1n != pos2n:
                            axs[idx_axs,idx_col].plot(
                                np.arange(pos1n,pos2n),
                                [np.amax(cnt_tmp[ pos1_:pos2_ ])*sign_now]*(pos2n-pos1n),
                                color="#808080"
                            )
                    cnt_tmp[ pos1_:pos2_ ] += 1
            
            ymax_tr = np.amax([ymax_tr,np.amax(cnt_tmp)])

        if ymax_tr<=20:
            axs[idx_axs,idx_col].set_ylim([-20-1,20+1])
            ymax_tr = 20
            yticks = list(range(-ymax_tr,ymax_tr+1,10))
            ylabels = [y if y>0 else -y for y in yticks]
        else:
            axs[idx_axs,idx_col].set_ylim([-ymax_tr-1,ymax_tr+1])
            ymax_tr = int(np.ceil((ymax_tr/10)*10))
            yticks = [-ymax_tr,0,ymax_tr]
            ylabels = [y if y>0 else -y for y in yticks]
            
        axs[idx_axs,idx_col].set_yticks(yticks)
        axs[idx_axs,idx_col].set_yticklabels(ylabels)
        axs[idx_axs,idx_col].set_ylabel('reads')

        tr_name = ref.id.dict_name[tr]['symbol']
        fig.suptitle(f'{tr} {tr_name}')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
    pdf.close()

    print("hoge")


def indv_plot2(
    save_dir,
    load_dir,
    ref,
    smpls,
    full_align=True,
    tr_list=[]
):
    save_dir = save_dir / 'indv_plot2'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    for c in ref.exp_metadata.df_metadata['condition'].unique():
        if c != 'control':
            continue
        
        cnt_pos_trs = {}
        for s in smpls:
            if ref.exp_metadata.df_metadata.loc[s,'condition'] != c:
                continue

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            if full_align:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
                idx_full = df_data['length'] == df_data['read_length']
                df_data = df_data.iloc[ idx_full.values, : ]
                if tr_list:
                    df_data = df_data[
                        np.sum([
                            (df_data['tr_id'] == tr).values
                            for tr in tr_list
                            ],axis=0).astype(bool)
                    ]
                dict_tr_now = list(df_data.groupby('tr_id'))
            else:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                dict_tr_now = list(df_data.groupby('tr_id'))

            tr_len = len(dict_tr_now)

            cnt_pos_tr = {}
            for i,(tr,df) in enumerate(dict_tr_now):
                print(f'\r{i}/{tr_len} transcripts...',end='')

                tmp = df['dist5_start'].values + ref.annot.annot_dict[tr]['start']
                pos,cnt = np.unique(tmp,return_counts=True)
                cnt_pos_tr[tr] = [pos,cnt]
            
            cnt_pos_trs[s] = cnt_pos_tr
        
        smpls_now = list(cnt_pos_trs.keys())
        tr_list = list(set(list(cnt_pos_trs[smpls_now[0]].keys())) &\
            set(list(cnt_pos_trs[smpls_now[1]].keys())) &\
            set(list(cnt_pos_trs[smpls_now[2]].keys())) &\
            set(list(cnt_pos_trs[smpls_now[3]].keys())))

        for i,tr in enumerate(tr_list):
            tr_name = ref.id.dict_name[tr]['symbol']
            
            if full_align:
                outfile_name = save_dir / f'{c}_{tr_name}.pdf'
            else:
                outfile_name = save_dir / f'all_{c}_{tr_name}.pdf'
            pdf = PdfPages(outfile_name)

            fig,axs = plt.subplots(
                4,1,figsize=(4,4),sharex=True,
                gridspec_kw={'height_ratios': [1,2,2,2]})
            axs[0].hlines(
                0,
                0,
                ref.annot.annot_dict[tr]['cdna_len'],
                color="#808080",
                lw=3
            )
            axs[0].hlines(
                0,
                ref.annot.annot_dict[tr]['start'],
                ref.annot.annot_dict[tr]['stop'],
                color="#880000",
                lw=10
            )
            axs[0].set_axis_off()
            for s in smpls_now:
                if ref.exp_metadata.df_metadata.loc[s,'pool'] == 'total':
                    idx_axs = 1
                else:
                    idx_axs = 2
                if ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense':
                    col_now = '#880000'
                    sign_now = 1
                else:
                    col_now = '#000088'
                    sign_now = -1

                # axs[idx_axs].hlines(
                #     0,-ref.annot.annot_dict[tr]['start'],
                #     ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'],
                #     color="#808080"
                # )

                axs[idx_axs].set_title(
                    ref.exp_metadata.df_metadata.loc[s,'pool']
                )
                if tr in cnt_pos_trs[s]:
                    axs[idx_axs].vlines(
                        cnt_pos_trs[s][tr][0],
                        0,
                        cnt_pos_trs[s][tr][1]*sign_now,
                        color=col_now,
                        lw=2
                    )
                
                axs[idx_axs].set_ylabel('reads')
            
            for ax in axs:
                ylim_now = [(y//5+1)*5+1 if y>0 else (y//5)*5-1 for y in ax.get_ylim()]
                ax.set_ylim(ylim_now)
                
                if ylim_now[0]>=-16:
                    ytick_neg = np.array([ylim_now[0]+1,0],dtype=int)
                elif ylim_now[0]>=-61:
                    ytick_neg = -np.arange(0,-(ylim_now[0]+1),20,dtype=int)[::-1]
                else:
                    ytick_neg = -np.arange(0,-(ylim_now[0]+1),50,dtype=int)[::-1]

                if ylim_now[1]<=16:
                    ytick_pos = np.array([ylim_now[1]-1],dtype=int)
                elif ylim_now[1]<=61:
                    ytick_pos = np.arange(20,ylim_now[1],20,dtype=int)
                else:
                    ytick_pos = np.arange(50,ylim_now[1],dtype=int)

                ax.set_yticks(np.hstack((ytick_neg,ytick_pos)))
                ax.set_yticklabels(np.hstack((-ytick_neg,ytick_pos)))

            fig.suptitle(f'{tr} {tr_name}')
            fig.tight_layout()
            fig.savefig(pdf,format='pdf')
            plt.close('all')
            plt.clf()
            pdf.close()


def indv_plot_multi2(
    save_dir,
    load_dir,
    ref,
    smpls,
    fname='',
    full_align=True
):
    save_dir = save_dir / 'indv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    cnt_pos_trs = {}
    cond_list = [];tr_list = []
    for s in smpls:
        cond = ref.exp_metadata.df_metadata.loc[s,'condition']
        strand = ref.exp_metadata.df_metadata.loc[s,'strand']
        pool = ref.exp_metadata.df_metadata.loc[s,'pool']

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        if full_align:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
            idx_full = df_data['length'] == df_data['read_length']
            df_data = df_data.iloc[ idx_full.values, : ]
            dict_tr_now = list(df_data.groupby('tr_id'))
        else:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
            dict_tr_now = list(df_data.groupby('tr_id'))

        tr_len = len(dict_tr_now)

        cnt_pos_tr = {}
        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            if len(df) < 8:
                continue

            tmp = df['dist5_start'].values + ref.annot.annot_dict[tr]['start']
            pos,cnt = np.unique(tmp,return_counts=True)
            cnt_pos_tr[tr] = [pos,cnt]
        
        cnt_pos_trs[(cond,pool,strand)] = cnt_pos_tr        
        tr_list = list(set(tr_list) | set(list(cnt_pos_tr.keys())))
        if cond not in cond_list:
            cond_list.append(cond)
    
    tr_list_names = [
        ref.id.dict_name[tr]['symbol']
        for tr in tr_list
    ]
    idx_sort = np.argsort(tr_list_names)
    tr_list = np.array(tr_list)[idx_sort]
    tr_list_names = np.array(tr_list_names)[idx_sort]
    dict_cond_col = dict(zip(cond_list,list(range(len(cond_list)))))

    if full_align:
        outfile_name = save_dir / f'indv_plot2{fname}.pdf'
    else:
        outfile_name = save_dir / f'indv_plot2_partial{fname}.pdf'

    # output the count data
    dict_out = {
        'number':np.arange(1,len(tr_list)+1),
        'transcript name':tr_list_names
    }
    for cond in cond_list:
        for pool in ['total','cap']:
            for strand in ['sense','antisense']:
                dict_out[ f'read count in {cond} {pool} {strand}' ] = [
                    np.sum(cnt_pos_trs[(cond,pool,strand)][tr][1])
                    if tr in cnt_pos_trs[(cond,pool,strand)] else 0
                    for tr in tr_list
                ]
    pd.DataFrame(dict_out,index=tr_list).to_csv(outfile_name.with_suffix('.csv.gz'))

    # plot    
    tr_len = len(tr_list)
    pdf = PdfPages(outfile_name)
    for i,tr in enumerate(tr_list):
        print(f'\r{i}/{tr_len} transcripts...',end='')
        tr_name = ref.id.dict_name[tr]['symbol']
        fig,axs = plt.subplots(
                3,len(cond_list),figsize=(4*len(cond_list),3),sharex=True,
                gridspec_kw={'height_ratios': [1,2,2]})
        for cond,pool,strand in cnt_pos_trs.keys():
            s = (cond,pool,strand)
            if pool == 'total':
                idx_axs = 1
            else:
                idx_axs = 2
            if strand == 'sense':
                col_now = '#880000'
                sign_now = 1
            else:
                col_now = '#000088'
                sign_now = -1
            idx_col = dict_cond_col[cond]

            # gene annotation
            if idx_axs == 1:
                axs[0,idx_col].hlines(
                    0,
                    0,
                    ref.annot.annot_dict[tr]['cdna_len'],
                    color="#808080",
                    lw=3
                )
                axs[0,idx_col].hlines(
                    0,
                    ref.annot.annot_dict[tr]['start'],
                    ref.annot.annot_dict[tr]['stop'],
                    color="#880000",
                    lw=10
                )
                
                axs[0,idx_col].text(
                    (ref.annot.annot_dict[tr]['stop']+ref.annot.annot_dict[tr]['start'])/2,
                    10,
                    '$\it{' + tr_name + '}$' if tr_name is str else '',
                    ha='center'
                )
                axs[0,idx_col].set_axis_off()

            # read plots
            axs[idx_axs,idx_col].set_title(f'{pool}, {cond}')
            if tr in cnt_pos_trs[s]:
                axs[idx_axs,idx_col].vlines(
                    cnt_pos_trs[s][tr][0],
                    0,
                    cnt_pos_trs[s][tr][1]*sign_now,
                    color=col_now,
                    lw=2
                )
            axs[idx_axs,idx_col].set_ylabel('reads')
            
        for ax in axs.flatten():
            ylim_now = [(y//5+1)*5+1 if y>0 else (y//5)*5-1 for y in ax.get_ylim()]
            ax.set_ylim(ylim_now)
            
            if ylim_now[0]>=-16:
                ytick_neg = np.array([ylim_now[0]+1,0],dtype=int)
            elif ylim_now[0]>=-61:
                ytick_neg = -np.arange(0,-(ylim_now[0]+1),20,dtype=int)[::-1]
            else:
                ytick_neg = -np.arange(0,-(ylim_now[0]+1),50,dtype=int)[::-1]

            if ylim_now[1]<=16:
                ytick_pos = np.array([ylim_now[1]-1],dtype=int)
            elif ylim_now[1]<=61:
                ytick_pos = np.arange(20,ylim_now[1],20,dtype=int)
            else:
                ytick_pos = np.arange(50,ylim_now[1],dtype=int)

            ax.set_yticks(np.hstack((ytick_neg,ytick_pos)))
            ax.set_yticklabels(np.hstack((-ytick_neg,ytick_pos)))
            
        fig.suptitle(f'{tr} {tr_name}')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
        plt.clf()
    pdf.close()


def indv_plot_ribo(
    save_dir,
    load_dir,
    ref,
    smpls,
    tr_list,
    fname
):
    save_dir = save_dir / 'indv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    tr_len = len(tr_list)
       
    reads_trs = {};pos_trs = {}
    for s in smpls:

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)
        dict_tr_now = list(df_data.groupby('tr_id'))

        reads_tr = {};pos_tr = {}
        for i,tr in enumerate(tr_list):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            df = dict_tr.get(tr,[])
            if len(df) == 0:
                continue
            reads = np.vstack(
                [
                    df['dist5_start'].values,
                    df['dist5_start'].values + df['length'].values,
                ]
            )
                
            pos = np.arange(
                start=-ref.annot.annot_dict[tr]['start'],
                stop=ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['start']
            )
            reads_tr[tr] = reads
            pos_tr[tr] = pos
        
        reads_trs[s] = reads_tr
        pos_trs[s] = pos_tr
        
    tr_list_names = [
        ref.id.dict_name[tr]['symbol']
        for tr in tr_list
    ]

    outfile_name = save_dir / f'{fname}.pdf'
    pdf = PdfPages(outfile_name)
    for i,tr in enumerate(tr_list):
        fig,axs = plt.subplots(len(smpls),1,figsize=(6,2*len(smpls)))
        for ii,s in enumerate(smpls):
            col_now = '#038701'
            
            cnt_tmp = np.ones(ref.annot.annot_dict[tr]['cdna_len']).astype(int)
            axs[ii].axvspan(
                0,ref.annot.annot_dict[tr]['cds_len'],
                color="#FFF8DC",alpha=0.5
            )
            axs[ii].hlines(
                0,-ref.annot.annot_dict[tr]['start'],
                ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'],
                color="#808080"
            )
            axs[ii].set_title(s)

            if tr in reads_trs[s]:
                for j in range(reads_trs[s][tr].shape[1]):
                    pos1 = reads_trs[s][tr][0,j]
                    pos2 = reads_trs[s][tr][1,j]
                    pos1_ = pos1 + ref.annot.annot_dict[tr]['start']
                    pos2_ = pos2 + ref.annot.annot_dict[tr]['start']
                    axs[ii].plot(
                        np.arange(pos1,pos2),
                        [np.amax(cnt_tmp[ pos1_:pos2_ ])]*(pos2_-pos1_),
                        color=col_now
                    )
                    
                    cnt_tmp[ pos1_:pos2_ ] += 1
            
            ymax = np.amax(cnt_tmp)
            if ymax<=20:
                axs[ii].set_ylim([-20-1,20+1])
                ymax = 20
                yticks = list(range(-ymax,ymax+1,10))
                ylabels = [y if y>0 else -y for y in yticks]
            else:
                axs[ii].set_ylim([-ymax-1,ymax+1])
                ymax = int(np.ceil((ymax/10)*10))
                yticks = [-ymax,0,ymax]
                ylabels = [y if y>0 else -y for y in yticks]
            
            axs[ii].set_yticks(yticks)
            axs[ii].set_yticklabels(ylabels)
            axs[ii].set_ylabel('reads')

        tr_name = ref.id.dict_name[tr]['symbol']
        fig.suptitle(f'{tr} {tr_name}')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
    pdf.close()

def indv_plot_ribo2(
    save_dir,
    load_dir,
    ref,
    smpls,
    tr_list
):
    save_dir = save_dir / 'indv_plot2'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    tr_len = len(tr_list)
       
    cnt_pos_trs = {};pos_trs = {}
    for s in smpls:

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)
        dict_tr_now = list(df_data.groupby('tr_id'))

        cnt_pos_tr = {};pos_tr = {}
        for i,tr in enumerate(tr_list):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            df = dict_tr.get(tr,[])
            if len(df) == 0:
                continue
            tmp = df['dist5_start'].values + ref.annot.annot_dict[tr]['start']
            pos,cnt = np.unique(tmp,return_counts=True)
                
            pos_ = np.arange(
                start=0,
                stop=ref.annot.annot_dict[tr]['cdna_len']
            )
            cnt_pos_tr[tr] = [pos,cnt]
            pos_tr[tr] = pos_
        
        cnt_pos_trs[s] = cnt_pos_tr
        pos_trs[s] = pos_tr

    for i,tr in enumerate(tr_list):
        tr_name = ref.id.dict_name[tr]['symbol']
        outfile_name = save_dir / f'indv_plot_{tr_name}.pdf'
        pdf = PdfPages(outfile_name)

        fig,axs = plt.subplots(
                len(smpls)+1,1,figsize=(4,1+len(smpls)),sharex=True,
                gridspec_kw={'height_ratios': [1]+[2]*len(smpls)})
        axs[0].hlines(
            0,
            0,
            ref.annot.annot_dict[tr]['cdna_len'],
            color="#808080",
            lw=3
        )
        axs[0].hlines(
            0,
            ref.annot.annot_dict[tr]['start'],
            ref.annot.annot_dict[tr]['stop'],
            color="#880000",
            lw=10
        )
        axs[0].set_axis_off()

        for ii,s in enumerate(smpls):
            col_now = '#038701'
            ii += 1

            # axs[ii].hlines(
            #     0,-ref.annot.annot_dict[tr]['start'],
            #     ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'],
            #     color="#808080"
            # )
            axs[ii].set_title(s)

            if tr in cnt_pos_trs[s]:
                axs[ii].vlines(
                    cnt_pos_trs[s][tr][0],
                    0,
                    cnt_pos_trs[s][tr][1],
                    color=col_now,
                    lw=2
                )
                    
                ylim_now = (axs[ii].get_ylim()[1]//5+1)*5+1
                axs[ii].set_ylim([-ylim_now*0.05,ylim_now])
                
                if ylim_now<=16:
                    ytick_pos = np.array([0,ylim_now-1],dtype=int)
                elif ylim_now<=61:
                    ytick_pos = np.arange(0,ylim_now,20,dtype=int)
                else:
                    ytick_pos = np.arange(0,ylim_now,100,dtype=int)

                axs[ii].set_yticks(ytick_pos)
                axs[ii].set_yticklabels(ytick_pos)
                axs[ii].set_ylabel('reads')

        fig.suptitle(f'{tr} {tr_name}')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
        pdf.close()

def detect_peaks(
    save_dir,
    load_dir,
    ref,
    smpls,
    thres_pos=8,
    full_align=True
):
    save_dir = save_dir / 'detect_peaks'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for s in smpls:

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        
        if full_align:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8)
            idx_full = df_data['length'] == df_data['read_length']
            df_data = df_data.iloc[ idx_full.values, : ]
            dict_tr_now = list(df_data.groupby('tr_id'))
        else:
            df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
            dict_tr_now = list(df_data.groupby('tr_id'))
        
        tr_len = len(dict_tr_now)

        peaks_pos = [];peaks_cnt = [];peaks_trs = [];regions = []
        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            pos_cnt = df.groupby('dist5_start').size()
            pos_cnt_thres = pos_cnt[ pos_cnt >= thres_pos ]
            if len(pos_cnt_thres)>0:
                peaks_pos += list(pos_cnt_thres.index)
                peaks_cnt += list(pos_cnt_thres.values)
                peaks_trs += [tr]*len(pos_cnt_thres)
                regions_tmp = np.array(['CDS']*len(pos_cnt_thres),dtype='<U4')
                pos_cnt_array = np.array(list(pos_cnt_thres.index))
                regions_tmp[ pos_cnt_array<0 ] = '5UTR'
                regions_tmp[ pos_cnt_array > ref.annot.annot_dict[tr]['stop'] ] = '3UTR'
                regions += list(regions_tmp)
        
        peaks_trs_names = [
            ref.id.dict_name[tr]['symbol']
            for tr in peaks_trs
        ]
        pd.DataFrame({
            'name':peaks_trs_names,
            'distance from start codon (nt)':peaks_pos,
            'count':peaks_cnt,
            'region':regions
        },index=peaks_trs
        ).to_csv(save_dir / f'peaks_{s}_thres{thres_pos}.csv.gz')

def _eval_overlap_peaks(
     df1,df2,pair,window   
):
    df1['dataset'] = pair[0]
    df2['dataset'] = pair[1]
    df1['dataset_org'] = pair[0]
    df2['dataset_org'] = pair[1]
    df_merge = pd.concat([df1,df2],axis=0)
    tr_uniq,cnt_uniq = np.unique(np.array(df_merge.index,dtype=str),return_counts=True)
    tr_uniq_ = tr_uniq[ cnt_uniq != 1 ]

    n_overlapped_now = {}
    n_overlapped_now[pair[0]] = []; n_overlapped_now[pair[1]] = []

    for tr in tr_uniq[ cnt_uniq == 1 ]:
        row = df_merge.loc[tr,:]
        if row['dataset'] == pair[0]:
            n_overlapped_now[pair[0]].append(0)
        elif row['dataset'] == pair[1]:
            n_overlapped_now[pair[1]].append(0)
        
    for tr in tr_uniq_:
        row = df_merge.loc[tr,:]
        if (pair[0] not in row['dataset'].values):
            if type(row) is pd.Series:
                n_overlapped_now[pair[1]].append(0)
            else:
                n_overlapped_now[pair[1]] += [0]*len(row)
            continue
        elif (pair[1] not in row['dataset'].values):
            if type(row) is pd.Series:
                n_overlapped_now[pair[0]].append(0)
            else:
                n_overlapped_now[pair[0]] += [0]*len(row)
            continue

        pos1 = row[ row['dataset'] == pair[0] ]['distance from start codon (nt)'].values
        pos2 = row[ row['dataset'] == pair[1] ]['distance from start codon (nt)'].values
        dataset_label_new = row['dataset'].values
        pos2_cnt = dict(zip(pos2,np.zeros(len(pos2),dtype=int)))
        for p1 in pos1:
            pos2_close = pos2[np.abs(pos2-p1) <= window]
            if len(pos2_close)>0:
                dataset_label_new[
                    (row['dataset']==pair[0])*\
                    (row['distance from start codon (nt)']==p1) ] = 'overlap'
                for p2 in pos2_close:
                    dataset_label_new[
                    (row['dataset']==pair[1])*\
                    (row['distance from start codon (nt)']==p2) ] = 'overlap'
                    pos2_cnt[p2] += 1
            n_overlapped_now[pair[0]].append(len(pos2_close))
        df_merge.loc[tr,'dataset'] = dataset_label_new

        for p2 in pos2:
            pos1_close = pos1[np.abs(pos1-p2) <= window]
            n_overlapped_now[pair[1]].append(len(pos1_close))

    df_cnt = df_merge.value_counts(['dataset'])
    df_cnt2 = df_merge.value_counts(['dataset','dataset_org'])

    n_overlapped_uniq1,n_overlapped_cnt1 = np.unique(n_overlapped_now[pair[0]],return_counts=True)
    n_overlapped_uniq2,n_overlapped_cnt2 = np.unique(n_overlapped_now[pair[1]],return_counts=True)
    assert np.sum(n_overlapped_uniq1*n_overlapped_cnt1) == np.sum(n_overlapped_uniq2*n_overlapped_cnt2)
    assert np.sum(n_overlapped_cnt1) == df_cnt[pair[0]] + df_cnt2[('overlap',pair[0])]
    assert np.sum(n_overlapped_cnt2) == df_cnt[pair[1]] + df_cnt2[('overlap',pair[1])]
    assert np.sum(n_overlapped_cnt1[n_overlapped_uniq1==0]) == df_cnt[pair[0]]
    assert np.sum(n_overlapped_cnt2[n_overlapped_uniq2==0]) == df_cnt[pair[1]]
    dict_n_overlapped = {
        pair[0]:dict(zip(n_overlapped_uniq1,n_overlapped_cnt1)),
        pair[1]:dict(zip(n_overlapped_uniq2,n_overlapped_cnt2))
    }

    return df_merge,df_cnt,df_cnt2,dict_n_overlapped

def _plot_overlap_peaks(
    outfile_name,
    labels,
    pairs,
    cnt_list,
    cnt_list2,
    n_overlapped,
    labels_plot  
):
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    
    idx_plot = 0
    for plot_type in ['venn','bar']:
        for pair in pairs:

            if plot_type == 'bar':
                fig,axs = plt.subplots(1,2,figsize=(4,3))
                
                b1 = axs[0].bar(
                    x=labels,
                    height=[cnt_list[pair][pair[0]],cnt_list[pair][pair[1]]],
                    label='Non-overlapped',
                    color="#3E3E3E"
                )
                b2 = axs[0].bar(
                    x=labels,
                    height=[cnt_list2[pair][('overlap',pair[0])],cnt_list2[pair][('overlap',pair[1])]],
                    label='overlapped',
                    bottom=[cnt_list[pair][pair[0]],cnt_list[pair][pair[1]]],
                    color="#BCBCBC"
                )
                for i,b_ in enumerate(b1):
                    axs[0].text(
                        b_.get_x()+b_.get_width()/2,
                        cnt_list[pair][pair[i]],
                        str(cnt_list[pair][pair[i]]),
                        ha='center'
                    )
                    axs[0].text(
                        b_.get_x()+b_.get_width()/2,
                        cnt_list[pair][pair[i]]+cnt_list2[pair][('overlap',pair[i])],
                        str(cnt_list2[pair][('overlap',pair[i])]),
                        ha='center'
                    )
                
                axs[0].set_ylabel('Number of peaks')
                
                axs[1].yaxis.set_visible(False)
                axs[1].xaxis.set_visible(False)
                for s in ['bottom','top','left','right']:
                    axs[1].spines[s].set_visible(False) 
                axs[1].legend(
                    handles=[b1,b2],
                    bbox_to_anchor=[1.1,1.1]
                )
                
            elif plot_type == 'venn':
                fig,axs = plt.subplots(1,3,figsize=(5,2))
                c = venn2(
                    subsets = [
                        cnt_list[pair][pair[0]],
                        cnt_list[pair][pair[1]],
                        cnt_list[pair]['overlap']
                    ],
                    set_labels=labels,
                    ax=axs[0]
                )
                c.get_patch_by_id('10').set_color('red')
                c.get_patch_by_id('01').set_color('blue')
                c.get_patch_by_id('11').set_color('magenta')
                c.get_patch_by_id('10').set_edgecolor('none')
                c.get_patch_by_id('01').set_edgecolor('none')
                c.get_patch_by_id('11').set_edgecolor('none')

                c = venn2(
                    subsets = [
                        cnt_list2[pair][(pair[0],pair[0])],
                        0,
                        cnt_list2[pair][('overlap',pair[0])],
                    ],
                    set_labels=(labels[0],''),
                    ax=axs[1]
                )
                c.get_patch_by_id('10').set_color('red')
                c.get_patch_by_id('01').set_color('blue')
                c.get_patch_by_id('11').set_color('magenta')
                c.get_patch_by_id('10').set_edgecolor('none')
                c.get_patch_by_id('01').set_edgecolor('none')
                c.get_patch_by_id('11').set_edgecolor('none')

                c = venn2(
                    subsets = [
                        0,
                        cnt_list2[pair][(pair[1],pair[1])],
                        cnt_list2[pair][('overlap',pair[1])],
                    ],
                    set_labels=('',labels[1]),
                    ax=axs[2]
                )
                c.get_patch_by_id('10').set_color('red')
                c.get_patch_by_id('01').set_color('blue')
                c.get_patch_by_id('11').set_color('magenta')
                c.get_patch_by_id('10').set_edgecolor('none')
                c.get_patch_by_id('01').set_edgecolor('none')
                c.get_patch_by_id('11').set_edgecolor('none')

            fig.suptitle(f'{pair[0]}, {pair[1]}')
            fig.tight_layout()
            plt.show()
            fig.savefig(pdf, format='pdf')
            fig.savefig(f'{outfile_name.parent}/{labels_plot[idx_plot]}_{outfile_name.stem}.png',dpi=200)
            idx_plot += 1
            plt.close()
    
    # how many reads are overlapped
    fig,axs = plt.subplots(2,1,figsize=(6,4))
    b = axs[0].bar(
        x=list(n_overlapped[pair][pair[0]].keys()),
        height=list(n_overlapped[pair][pair[0]].values()),
        color="#3E3E3E"
    )
    for i,(k,v) in enumerate(n_overlapped[pair][pair[0]].items()):
        axs[0].text(
            b[i].get_x()+b[i].get_width()/2,
            b[i].get_height(),
            str(v),
            ha='center'
        )
    axs[0].set_xlabel(f'Number of overlap with {pair[1]} peaks')
    axs[0].set_title(pair[0])
    axs[1].set_ylabel('Counts')

    b = axs[1].bar(
        x=list(n_overlapped[pair][pair[1]].keys()),
        height=list(n_overlapped[pair][pair[1]].values()),
        color="#3E3E3E"
    )
    for i,(k,v) in enumerate(n_overlapped[pair][pair[1]].items()):
        axs[1].text(
            b[i].get_x()+b[i].get_width()/2,
            b[i].get_height(),
            str(v),
            ha='center'
        )
    axs[1].set_title(pair[1])
    axs[1].set_xlabel(f'Number of overlap with {pair[0]} peaks')
    axs[1].set_ylabel('Counts')
    fig.tight_layout()
    plt.show()
    fig.savefig(pdf, format='pdf')
    fig.savefig(f'{outfile_name.parent}/{labels_plot[idx_plot]}_{outfile_name.stem}.png',dpi=200)
    plt.close()

    pdf.close()

def overlap_peaks(
    save_dir,
    load_dir,
    ref,
    pairs,
    window:int,
    thres_pos:int
):
    save_dir = save_dir / 'overlap_peaks'
    if not save_dir.exists():
        save_dir.mkdir()

    cnt_list = {};cnt_list2 = {}
    n_overlapped = {}
    for pair in pairs:
        df1 = pd.read_csv(load_dir / f'peaks_{pair[0]}_thres{thres_pos}.csv.gz',index_col=0,header=0)
        df2 = pd.read_csv(load_dir / f'peaks_{pair[1]}_thres{thres_pos}.csv.gz',index_col=0,header=0)
        df_merge,df_cnt,df_cnt2,dict_n_overlapped = _eval_overlap_peaks(df1,df2,pair,window)
        cnt_list[pair] = df_cnt
        cnt_list2[pair] = df_cnt2
        n_overlapped[pair] = dict_n_overlapped

        df_merge.to_csv(save_dir / f'df_overlap_peaks_{pair[0]}_{pair[1]}_{thres_pos}_win{window}.csv.gz')
    
    outfile_name = save_dir / f'overlap_peaks_{thres_pos}_win{window}'
    _plot_overlap_peaks(
        outfile_name,
        ['Total','Capped'],
        pairs,
        cnt_list,
        cnt_list2,
        n_overlapped,
        ['venn_sense','venn_antisense','bar_sense','bar_antisense','bar-multiple']  
    )
    
    
def _calc_dendrogam(X,n_cluster):
    model = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward',compute_distances=True)
    model.fit_predict(X)
    # create counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i,merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 #leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return model, linkage_matrix 

def ribo_density_peaks(
    save_dir,
    load_dir,
    ref,
    smpls,
    df_peaks_path,
    fname
):
    save_dir = save_dir / 'ribo_density_peaks'
    if not save_dir.exists():
        save_dir.mkdir()
    
    df_peaks = pd.read_csv(df_peaks_path,header=0).set_axis(
        ['tr','tr_name','pos','cnt'],axis=1
    )
    df_peaks_tr = list(df_peaks.groupby('tr'))
    tr_len = len(df_peaks_tr)

    regions = {
        '-240 to -180':[-240,-180],
        '-180 to -120':[-180,-120],
        '-120 to -60':[-120,-60],
        '-60 to 0':[-60,0],
        '0 to +60':[0,60],
        '+60 to +120':[60,120],
        '+120 to +180':[120,180],
        '+180 to +240':[180,240],
    }
    
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)
        
        cnt_mat = []
        num_tr = 0
        for i,(tr,df) in enumerate(df_peaks_tr):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            if tr not in dict_tr.keys():
                continue

            if len(df)>1:
                pos_peak = df[ df['cnt'] == df['cnt'].max() ]['pos'].iloc[0]
            else:
                pos_peak = df['pos'].iloc[0]
            
            df_ribo_now = dict_tr[tr].groupby('dist5_start').size()

            cnts = []
            for k,v in regions.items():
                cnt = len(df_ribo_now[
                    (df_ribo_now.index >= (v[0] + pos_peak)) * (df_ribo_now.index < (v[1] + pos_peak))
                    ])
                cnts.append(cnt)
            
            if np.sum(cnts) < 16:
                continue

            cnt_mat.append(cnts)
            num_tr += 1
        
        # box plot
        pdf = PdfPages(save_dir / f'box_ribo_density_peaks_{fname}_{s}.pdf')
        fig,ax = plt.subplots(1,1,figsize=(len(regions),3))

        cnt_tmp = [[]]*len(regions)
        for i in range(len(regions)):
            cnt_tmp[i] = [v[i] for v in cnt_mat]

        ax.boxplot(
            x=cnt_tmp,
            labels=[k for k in regions.keys()],
            showfliers=False,
            boxprops={'facecolor':"#808080"},
            whiskerprops={'color':"#000000"},
            medianprops={'color':"#FF0000"},
            capprops={'color':"#000000"},
            patch_artist=True)
        ax.set_ylabel('reads')
        ax.set_xlabel('distance from Cap-antisense peaks (nt)')
        fig.suptitle(s)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        pdf.close()
        

        # line plot
        pdf = PdfPages(save_dir / f'line_ribo_density_peaks_{fname}_{s}.pdf')
        fig,ax = plt.subplots(1,1,figsize=(len(regions),3))

        num_region = len(regions)
        for i in range(num_tr):
            ax.plot(
                list(range(num_region)),
                cnt_mat[i],
                marker='.',
                color='#808080'
            )
        ax.set_xticks(list(range(num_region)))
        ax.set_xticklabels(list(regions.keys()))
        ax.set_ylabel('reads')
        ax.set_xlabel('distance from Cap-antisense peaks (nt)')
        fig.suptitle(s)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        pdf.close()

        # heatmap
        n_cluster = 3
        model_tr,linkage_tr = _calc_dendrogam(np.array(cnt_mat),n_cluster)

        pdf = PdfPages(save_dir / f'heatmap_ribo_density_peaks_{fname}_{s}.pdf')
        fig, axs = plt.subplots(1, 3, figsize=(6,10),gridspec_kw={'width_ratios': [10,1,1]})
        X = ( np.array(cnt_mat) - np.mean(np.array(cnt_mat),axis=1)[:,np.newaxis] ) / np.std( np.array(cnt_mat),axis=1 )[:,np.newaxis]
        sns.heatmap(
            X,
            cmap="coolwarm",
            vmin=-4,vmax=4,
            cbar_kws={'label':'reads','fraction':0.4},
            cbar_ax=axs[2],
            yticklabels=False,
            xticklabels=list(regions.keys()),
            ax=axs[0])
        axs[0].set_xticklabels(list(regions.keys()),rotation=30,ha="right")
        
        cmap = mpl.cm.get_cmap('Set3')
        start = 0
        for i in range(n_cluster):
            d = X[model_tr.labels_ == i,:]
            stop = start + len(d)
            axs[1].fill_between(
                y1=start,y2=stop,
                x=[0,1],
                color=cmap( i/n_cluster )
            )
            start = stop
        axs[1].sharey(axs[0])
        axs[1].set_xticks([])
        axs[1].set_title("Clusters")
        # axs[1].set_yticks([])
        for tmp in ['top','right','bottom','left']:
            axs[1].spines[tmp].set_visible(False)

        fig.tight_layout()
        fig.savefig(pdf,format='pdf')

        # each cluster information
        fig, axs = plt.subplots(1, n_cluster, figsize=(n_cluster*2,3))
        for i,ax in enumerate(axs):
            d = X[model_tr.labels_ == i,:]
            mean_grps = np.mean(d,axis=0)
            std_grps = np.std(d,axis=0)
            if d.shape[1]==1:
                std_grps = np.zeros(d.shape[0])
            axs[i].plot(
                list(range(num_region)),
                mean_grps,
                marker='.',
                color='#808080'
            )
            axs[i].fill_between(
                list(range(num_region)),
                mean_grps-std_grps,mean_grps+std_grps,
                color='#808080',
                alpha=0.5
            )
            axs[i].set_xticklabels(
                axs[i].get_xticklabels(), 
                rotation=30, ha='right',fontsize=7.5)
            axs[i].set_title(f'cluster {i}\n(n={d.shape[0]})',fontsize=7.5)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')

        pdf.close()
        plt.close('all')

def sequence_peaks(
    save_dir,
    load_dir,
    ref,
    smpls,
    thres_pos=8,
    offset=50
):
    save_dir = save_dir / 'sequence_peaks'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for s in smpls:
        df_peaks = pd.read_csv(load_dir / f'peaks_{s}_thres{thres_pos}.csv.gz',index_col=0,header=0)\
            .set_axis(['tr_name','pos','cnt'],axis=1)
        with gzip.open(save_dir / f'seq_peaks_{s}_offset{offset}.fasta.gz','wt') as f:
            for tr,row in df_peaks.iterrows():
                pos = row['pos']
                seq = ref.ref_cdna[tr][
                    row['pos']-offset+ref.annot.annot_dict[tr]['start'] \
                    : row['pos']+offset+ref.annot.annot_dict[tr]['start'] ]
                if len(seq) == offset*2:
                    f.write(f'>{tr}_{pos}\n')
                    f.write(f'{seq}\n')

def sequence_trans(
    save_dir,
    load_dir,
    ref,
    smpls,
    tr_list,
    full_align=True
):
    save_dir = save_dir / 'sequence_trans'
    if not save_dir.exists():
        save_dir.mkdir()
    tr_len = len(tr_list)

    offset = 15

    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])

    for s in smpls:

        if (ref.exp_metadata.df_metadata.loc[s,'strand'] == 'antisense') and\
            (ref.exp_metadata.df_metadata.loc[s,'condition'] == 'control') and\
            (ref.exp_metadata.df_metadata.loc[s,'pool'] == 'cap'):

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            if full_align:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                # idx_full_ = df_data['length'] == df_data['read_length']
                idx_full = df_data['dist3_start'] == (df_data['read_length']+df_data['dist5_start']-1)
                df_data = df_data.iloc[ idx_full.values, : ]
                dict_tr_now = dict(list(df_data.groupby('tr_id')))
            else:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                dict_tr_now = dict(list(df_data.groupby('tr_id')))

            if full_align:
                outfile_name = save_dir / f'sequence_{s}.txt.gz'
            else:
                outfile_name = save_dir / f'sequence_all_{s}.txt.gz'
            
            with gzip.open(outfile_name,'wt') as f:
            
                for i,tr in enumerate(tr_list):
                    print(f'\r{i}/{tr_len} transcripts...',end='')

                    df = dict_tr_now.get(tr,[])
                    if len(df) == 0:
                        continue
                    tr_name = ref.id.dict_name[tr]['symbol']
                    f.write(f'>{tr}_{tr_name}\n')

                    df_grp = list(df.groupby(['dist5_start','length']))
                    
                    for (pos,length),df_now in df_grp:
                        cnt = len(df_now)
                        if np.any(df_now['length'] != df_now['read_length']):
                            print('error')
                        seq = my._rev(df_now['seq'].iloc[0],is_rev=False)
                        seq_region = ref.ref_cdna[tr][
                            pos-offset+ref.annot.annot_dict[tr]['start'] \
                            : pos+length+offset+ref.annot.annot_dict[tr]['start'] ]
                        
                        if len(seq_region) != (offset*2+length):
                            nt_5end = -(pos-offset+ref.annot.annot_dict[tr]['start'])
                            nt_3end = (pos+length+offset+ref.annot.annot_dict[tr]['start']) -\
                                (ref.annot.annot_dict[tr]['cdna_len']-ref.annot.annot_dict[tr]['start'])
                            if nt_5end>0:
                                seq_5end = ' '*nt_5end
                                pos_5end = 0
                            else:
                                seq_5end = ''
                                pos_5end = pos-offset+ref.annot.annot_dict[tr]['start']
                            if nt_3end>0:
                                seq_3end = ' '*nt_3end
                                pos_3end = ref.annot.annot_dict[tr]['cdna_len']
                            else:
                                seq_3end = ''
                                pos_3end = pos+length+offset+ref.annot.annot_dict[tr]['start']
                            seq_region = seq_5end + ref.ref_cdna[tr][pos_5end:pos_3end] + seq_3end
                            
                        
                        f.write('\t'+str(pos-offset)+' '*(offset*2+length-len(str(pos-offset))-1)+str(pos+offset+length)+'\n')
                        f.write('\t'+seq_region+'\n')
                        f.write('\t'+'-'*offset+seq+'-'*offset+'\n')
                        f.write('\t'+' '*(offset)+str(pos)+' '*(length-len(str(pos))-1)+str(pos+length)+'\n')
                        f.write(f'\t{length}\t{cnt}\n')

def select_cap_total_trs(
    save_dir,
    load_dir,
    ref,
    smpls,
    thres_pos,
    full_align=True
):
    save_dir = save_dir / 'select_cap_total_trs'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    for s in smpls:

        if (ref.exp_metadata.df_metadata.loc[s,'strand'] == 'sense') and\
            (ref.exp_metadata.df_metadata.loc[s,'condition'] == 'control') and\
            (ref.exp_metadata.df_metadata.loc[s,'pool'] == 'cap'):

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            if full_align:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)
                # idx_full_ = df_data['length'] == df_data['read_length']
                idx_full = df_data['dist3_start'] == (df_data['read_length']+df_data['dist5_start']-1)
                df_data = df_data.iloc[ idx_full.values, : ]
                dict_tr_now = dict(list(df_data.groupby('tr_id')))
            else:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                dict_tr_now = dict(list(df_data.groupby('tr_id')))
            
            tr_list = [];cnt_list = []
            tr_len = len(dict_tr_now)
            for i,(tr,df) in enumerate(dict_tr_now.items()):
                print(f'\r{i}/{tr_len} transcripts...',end='')
                
                pos5_cnt = df.groupby('dist5_start').size()
                pos5_cnt = pos5_cnt.set_axis(pos5_cnt.index + ref.annot.annot_dict[tr]['start'],axis=0)
                cnt = pos5_cnt[ pos5_cnt.index < thres_pos ].sum()
                if cnt > 8:
                    tr_list.append(tr)
                    cnt_list.append(cnt)

            pd.DataFrame(
                {
                    'count':cnt_list
                },index=tr_list
            ).to_csv(save_dir / f'tr_list_thres{thres_pos}nt_{s}.csv.gz')     

def select_trs_with_peaks(
    save_dir,
    load_dir,
    ref,
    smpls,
    thres_pos:list,
    full_align=True
):
    """
    thres_pos = ['dist5_start, 10]
    """
    save_dir = save_dir / 'select_trs_with_peaks'
    if not save_dir.exists():
        save_dir.mkdir()
    
    try:
        ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    except:
        print("exp_metadata has been already updated.")
    
    for s in smpls:

        if (ref.exp_metadata.df_metadata.loc[s,'strand'] == 'antisense') and\
            (ref.exp_metadata.df_metadata.loc[s,'condition'] == 'control'):

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            if full_align:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)
                # idx_full_ = df_data['length'] == df_data['read_length']
                idx_full = df_data['dist3_start'] == (df_data['read_length']+df_data['dist5_start']-1)
                df_data = df_data.iloc[ idx_full.values, : ]
                dict_tr_now = dict(list(df_data.groupby('tr_id')))
            else:
                df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=True)
                dict_tr_now = dict(list(df_data.groupby('tr_id')))
            
            tr_list = [];cnt_list = []
            tr_len = len(dict_tr_now)
            for i,(tr,df) in enumerate(dict_tr_now.items()):
                print(f'\r{i}/{tr_len} transcripts...',end='')
                
                pos5_cnt = df.groupby(thres_pos[0]).size()
                if (len(thres_pos) == 2) and (thres_pos[0] == 'dist5_start'):
                    cnt = pos5_cnt[
                        (pos5_cnt.index >= 0) *\
                        (pos5_cnt.index < thres_pos[1])].sum()
                    fname = f'tr_list_{thres_pos[0]}_thres{thres_pos[1]}nt_{s}.csv.gz'
                elif (len(thres_pos) == 2) and (thres_pos[0] == 'dist5_stop'):
                    cnt = pos5_cnt[
                        (pos5_cnt.index <= 0) *\
                        (pos5_cnt.index < thres_pos[1])].sum()
                    fname = f'tr_list_{thres_pos[0]}_thres{thres_pos[1]}nt_{s}.csv.gz'
                elif len(thres_pos) == 3:
                    cnt = pos5_cnt[
                        (pos5_cnt.index >= thres_pos[1]) *\
                        (pos5_cnt.index < thres_pos[2]) ].sum()
                    fname = f'tr_list_{thres_pos[0]}_thres{thres_pos[1]}-{thres_pos[2]}nt_{s}.csv.gz'
                if cnt > 8:
                    tr_list.append(tr)
                    cnt_list.append(cnt)

            pd.DataFrame(
                {
                    'count':cnt_list
                },index=tr_list
            ).to_csv(save_dir / fname)     

def ribo_density_trs(
    save_dir,
    load_dir,
    ref,
    smpls,
    tr_list,
    fname
):
    save_dir = save_dir / 'ribo_density_trs'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)

        tr_len = len(dict_tr)

        norm_cnt = [];norm_cnt_bg = []
        for i,(tr,df) in enumerate(dict_tr.items()):
            print(f'\r{i}/{tr_len} transcripts...',end='')
            if len(df) <= 8:
                continue
            norm_cnt_now = len(df.iloc[(df['cds_label']==1).values,:]) / ref.annot.annot_dict[tr]['cds_len'] * 100

            if tr in tr_list:
                norm_cnt.append(norm_cnt_now)
            else:
                norm_cnt_bg.append(norm_cnt_now)
            
        # box plot
        pdf = PdfPages(save_dir / f'box_ribo_density_trs_{fname}_{s}.pdf')
        fig,ax = plt.subplots(1,1,figsize=(3,3))

        labels=[
            f'Transcript with peaks\n(n={len(norm_cnt)})',
            f'Transcript without peaks\n(n={len(norm_cnt_bg)})'
        ]

        ax.boxplot(
            x=[norm_cnt,norm_cnt_bg],
            labels=labels,
            showfliers=False,
            boxprops={'facecolor':"#808080"},
            whiskerprops={'color':"#000000"},
            medianprops={'color':"#FF0000"},
            capprops={'color':"#000000"},
            patch_artist=True)
        # ax.set_xticks()
        ax.set_xticklabels(labels,rotation=30)
        ax.set_ylabel('Average ribosome density')
        fig.suptitle(s)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        pdf.close()

def ribo_density_trs_bins(
    save_dir,
    load_dir,
    ref,
    smpls,
    dict_tr_lists,
    fname,
    cumulative=True
):
    save_dir = save_dir / 'ribo_density_trs_bins'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_seq=False)

        tr_len = len(dict_tr)

        norm_cnts = {};norm_cnt = [];norm_cnt_bg = []
        tr_list_all = []
        for tr_list in dict_tr_lists.values():
            tr_list_all += list(tr_list)
        
        for tr_list_name,tr_list in dict_tr_lists.items():
            if not cumulative:
                norm_cnt = []
            for tr in tr_list:
                df = dict_tr.get(tr,[])
                if len(df)>8:
                    norm_cnt.append(
                        len(df.iloc[(df['cds_label']==1).values,:]) / ref.annot.annot_dict[tr]['cds_len'] * 100
                    )

            tr_list_name_now = f'{tr_list_name} (n={len(norm_cnt)})'
            norm_cnts[tr_list_name_now] = norm_cnt.copy()
        
        norm_cnt_bg = [
            len(dict_tr[tr].iloc[(dict_tr[tr]['cds_label']==1).values,:]) / ref.annot.annot_dict[tr]['cds_len'] * 100
            for tr in dict_tr.keys()
            if (tr not in tr_list_all) and (len(dict_tr[tr])>8)
        ]
            
        # box plot
        pdf = PdfPages(save_dir / f'box_ribo_density_trs_{fname}_{s}.pdf')
        if len(dict_tr_lists)>5:
            fig,ax = plt.subplots(1,1,figsize=(len(dict_tr_lists)*0.5,3))
        else:
            fig,ax = plt.subplots(1,1,figsize=(len(dict_tr_lists),3))

        norm_cnts[f'no peaks (n={len(norm_cnt_bg)})'] = norm_cnt_bg
        ax.hlines(
            y=np.median(norm_cnt_bg),
            xmin=0,
            xmax=len(norm_cnts)+1,
            linestyles='dashed',
            colors='#1975B5'
        )
        ax.boxplot(
            x=norm_cnts.values(),
            labels=np.arange(0,len(norm_cnts)),
            showfliers=False,
            boxprops={'facecolor':"#808080"},
            whiskerprops={'color':"#000000"},
            medianprops={'color':"#FF0000"},
            capprops={'color':"#000000"},
            patch_artist=True)
        # # scatters
        # for i,norm_cnt in enumerate(norm_cnts.values()):
        #     if i == (len(norm_cnts)-1):
        #         continue
        #     ii = ax.get_xticks()[i]
        #     ax.scatter(
        #         x=[ii]*len(norm_cnt),
        #         y=norm_cnt,
        #         s=1,
        #         color='#000000'
        #     )
        # ax.set_xticks()
        ax.set_xticklabels(
            norm_cnts.keys(),rotation=30,
            ha='right')
        ax.set_xlim([-0.5,len(norm_cnts)+0.5])
        ax.set_ylabel('Average ribosome density')
        fig.suptitle(s)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        pdf.close()
 
def detect_peaks_GTI(
    save_dir,
    load_dir,
    ref,
    smpls_LTM,
    smpls_CHX
):
    save_dir = save_dir / 'detect_peaks_GTI'
    if not save_dir.exists():
        save_dir.mkdir()
    
    from scipy.signal import find_peaks

    dict_tr_CHX = {}
    for s in smpls_CHX:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=10,is_seq=False)
        dict_tr_CHX[s] = dict_tr
    
    for s in smpls_LTM:

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1,is_seq=False) 
        tr_len = len(dict_tr)

        peaks_pos = [];peaks_cnt = [];peaks_trs = []
        for i,(tr,df) in enumerate(dict_tr.items()):
            print(f'\r{i}/{tr_len} transcripts...',end='')

            if len(df)<10:
                continue

            for d in dict_tr_CHX.values():
                if tr not in d.keys():
                    continue
                pos_cnt_GTI = df.groupby('dist5_start').size()
                pos_cnt_CHX = d[tr].groupby('dist5_start').size()

                x = np.arange(pos_cnt_GTI.index[0],pos_cnt_GTI.index[-1]+1,dtype=int)
                cnt = np.zeros_like(x)
                cnt[ np.array([np.where(x == i)[0][0] for i in pos_cnt_GTI.index]) ] = pos_cnt_GTI.values

                local_max_idx,_ = find_peaks(cnt,height=10,distance=7)
                if len(local_max_idx)>0:
                    for i in local_max_idx:
                        if x[i] not in pos_cnt_CHX.index:
                            continue
                        r_LTM = (cnt[i]/len(pos_cnt_GTI)) * 10
                        r_CHX = (pos_cnt_CHX[x[i]]/len(pos_cnt_CHX)) * 10
                        if (r_LTM - r_CHX)>0.05:
                            peaks_pos.append(x[i])
                            peaks_cnt.append(cnt[i])
                            peaks_trs.append(tr)
        
        peaks_trs_names = [
            ref.id.dict_name[tr]['symbol']
            for tr in peaks_trs
        ]
        df = pd.DataFrame({
            'name':peaks_trs_names,
            'distance from start codon (nt)':peaks_pos,
            'count':peaks_cnt
        },index=peaks_trs)
        df = df.drop_duplicates()
        df.to_csv(save_dir / f'peaks_{s}.csv.gz')


def overlap_peaks_GTI(
    save_dir,
    load_dir_trans,
    load_dir_GTI,
    ref,
    pairs,
    window:int,
    thres_pos:int
):
    save_dir = save_dir / 'overlap_peaks_GTI'
    if not save_dir.exists():
        save_dir.mkdir()

    cnt_list = {};cnt_list2 = {}
    n_overlapped = {}
    for pair in pairs:
        df1 = pd.read_csv(load_dir_trans / f'peaks_{pair[0]}_thres{thres_pos}.csv.gz',index_col=0,header=0)
        df2 = pd.read_csv(load_dir_GTI / f'peaks_{pair[1]}.csv.gz',index_col=0,header=0)
        df_merge,df_cnt,df_cnt2,dict_n_overlapped = _eval_overlap_peaks(df1,df2,pair,window)
        cnt_list[pair] = df_cnt
        cnt_list2[pair] = df_cnt2
        n_overlapped[pair] = dict_n_overlapped

        df_merge.to_csv(save_dir / f'df_overlap_peaks_{pair[0]}_{pair[1]}_{thres_pos}_win{window}.csv.gz')

    outfile_name = save_dir / f'overlap_peaks_{thres_pos}_win{window}'
    _plot_overlap_peaks(
        outfile_name,
        ['NAT','GTI'],
        pairs,
        cnt_list,
        cnt_list2,
        n_overlapped,
        ['venn_NAT','bar_NAT','bar-multiple']  
    )

def pca(
    save_dir:Path,
    path_cnt_data:Path,
    ref:my.Ref,
    color_markers:dict
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    save_dir = save_dir / 'pca'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    df_plot = pd.read_csv(path_cnt_data,index_col=0,header=0)
    X = df_plot.iloc[:,2:].values.T
    tr_list = list(df_plot.index)
    smpls = [c.replace('read count in ','') for c in df_plot.columns[2:]]
        
    outfile_name = save_dir / f'pca.pdf'
    
    # PCA
    pdf = PdfPages(outfile_name)
    for k in range(2):
        if k == 0:
            # all the transcripts
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
            X_pca = pipeline.fit_transform(X)
        elif k == 1:
            # only transcripts with >0 in all the dataset
            X_ = X[ :, np.all(X>0,axis=0) ]
            X_pca = pipeline.fit_transform(X_)

        explained_variance = pipeline['pca'].explained_variance_ratio_*100

        fig,axs = plt.subplots(1,2,figsize=(9,3))

        for i,(pc1,pc2) in enumerate([(0,1),(1,2)]):
            for j,s in enumerate(smpls):
                axs[i].scatter(
                    X_pca[j,pc1],X_pca[j,pc2],
                    color=color_markers[s][0],
                    marker=color_markers[s][1],
                    label=s)
            axs[i].set_xlabel(f'PC{pc1+1} ({round(explained_variance[pc1],2)}%)')
            axs[i].set_ylabel(f'PC{pc2+1} ({round(explained_variance[pc2],2)}%)')
        axs[1].legend(bbox_to_anchor=[1.1,1.1])
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
        plt.clf()
    pdf.close()
    print("hoge")

    