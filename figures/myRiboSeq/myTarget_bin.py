from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import matplotlib.style as mplstyle
import gzip

import myRiboSeq.mylib_bin as my
import myRiboSeq.myRiboBin as mybin

color_frame = my.color_frame
codon_table = my.codon2aa_table

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def indiv_plot(
    save_dir,
    load_dir,
    tr_list,
    fname,
    mode,
    ref,
    smpls,
    tr_list_names=[]
):
    save_dir = save_dir / f'indiv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    num_smpl = len(smpls)
    dts_cds = {};dts_5utr={};dts_3utr={}
    for p in smpls:
        if mode == 'coverage':
            dts_cds[p] = my._myload(load_dir / f'norm_cvgs_cds_{p}.joblib')
            dts_5utr[p] = my._myload(load_dir / f'norm_cvgs_5utr_{p}.joblib')
            dts_3utr[p] = my._myload(load_dir / f'norm_cvgs_3utr_{p}.joblib')
        elif 'offset' in mode:
            offset = int(mode.split('_')[-1])
            offset = int(mode.split('_')[-1])
            obj = mybin.myBinRiboNorm(
                smpl=p,
                sp=ref.sp,
                save_dir=load_dir,
                mode='start',
                read_len_list=[],
                is_length=False,
                is_norm=True,
                dirname=''
            )
            obj.decode()
            tmp = pd.DataFrame(obj.count['count5'].T.todense(),index=obj.tr['tr5'],columns=obj.pos['pos5'])
            dt = tmp.iloc[ tmp.index.map(lambda x: x in tr_list), : ]
            dt = dt.set_axis( dt.columns+offset, axis='columns', copy=False )
            dt = dt[ dt < 1000 ]
            dts_cds[p] = {};dts_5utr[p]={};dts_3utr[p]={}
            for tr in tr_list:
                if tr not in dt.index:
                    continue
                len_5utr = ref.annot.annot_dict[tr]['start']
                len_cds = ref.annot.annot_dict[tr]['cds_len']
                len_3utr = ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['stop'] - 3
                len_tr = ref.annot.annot_dict[tr]['cdna_len']
                if len_5utr > 0:
                    cols = np.array(range(-len_5utr,0))
                    dts_5utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                        fill_value=0
                    )
                else:
                    dts_5utr[p][tr] = []

                cols = np.array(range(0,len_cds))
                dts_cds[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                    dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                    fill_value=0
                )
                # dts_5utr[tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
                # dts_cds[tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
                if len_3utr <= 0:
                    dts_3utr[p][tr] = []
                else:
                    cols = np.array(range(len_cds,len_tr-len_5utr))
                    dts_3utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                        fill_value=0
                    )

    outfile_name = save_dir / f'indiv_plot_{fname}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        fig, ax = plt.subplots(num_smpl,1,figsize=(10,1.5*num_smpl),sharex=True,sharey=True)
        ax[-1].set_xlabel('Position along the transcript (nt)') 
        ax[-1].set_ylabel('Normalized density')
        tr_name = ref.id.dict_name[tr]['symbol']
        if len(tr_list_names)>0:
            fig.suptitle(f'{tr_name}_{tr}\n{tr_list_names[ii]}')
        else:
            fig.suptitle(f'{tr_name}_{tr}')
        # seq = ref.ref_cdna[tr][
        #     ref.annot.annot_dict[tr]["start"] : ref.annot.annot_dict[tr]["stop"]+3
        # ]
        if mode == 'coverage':
            lw_tmp = 1000/ref.annot.annot_dict[tr]['cdna_len']
        elif mode == 'asite':
            lw_tmp = 1
        else:
            lw_tmp = 1
        color = "#818589"
        xs = {};ys = {}
        for iii,p in enumerate(smpls):
            if tr not in dts_5utr[p]:
                continue
            ys[p] = np.hstack((dts_5utr[p][tr],dts_cds[p][tr],dts_3utr[p][tr]))
            xs[p] = np.arange( -len(dts_5utr[p][tr]),len(dts_3utr[p][tr])+len(dts_cds[p][tr]), 1, dtype=int )
            ax[iii].axvspan(0,len(dts_cds[p][tr]),color="#FFF8DC",alpha=0.5)
            if tr == 'ENST00000674920':#ATF4:
                # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3
                # uORF1 
                ax[iii].axvspan(87-len(dts_5utr[p][tr]),98-len(dts_5utr[p][tr]),color="#7393B3",alpha=0.5)
                ax[iii].axvspan(186-len(dts_5utr[p][tr]),365-len(dts_5utr[p][tr]),color="#FF69B4",alpha=0.5)
            
            ax[iii].vlines(xs[p],0,ys[p],colors=color,lw=lw_tmp)
            ax[iii].set_title(p)
        
        ylim_now = np.max([ a.get_ylim()[1] for a in ax])
        ax[0].set_ylim(ax[0].get_ylim()[0],ylim_now)
                    
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close()
    pdf.close()

def indiv_plot_smpl(
    save_dir,
    load_dir,
    tr_list,
    tr_list_names,
    fname,
    mode,
    ref,
    smpl,
    read_len_list=[],
    pos_plot=[]
):
    save_dir = save_dir / f'indiv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    dts_cds = {};dts_5utr={};dts_3utr={}
    p = smpl
    if mode == 'coverage':
        dts_cds[p] = my._myload(load_dir / f'norm_cvgs_cds_{p}.joblib')
        dts_5utr[p] = my._myload(load_dir / f'norm_cvgs_5utr_{p}.joblib')
        dts_3utr[p] = my._myload(load_dir / f'norm_cvgs_3utr_{p}.joblib')
    elif 'offset' in mode:
        offset = int(mode.split('_')[-1])
        if len(read_len_list)>0:
            dt = my._myload(load_dir / f'df_norm_density_start_{p}_{read_len_list[0]}-{read_len_list[-1]}.pkl')
        else:
            dt = my._myload(load_dir / f'df_norm_density_start_{p}.pkl')
        tmp = pd.DataFrame(dt['count5'].T.todense(),index=dt['tr5'],columns=dt['pos5'])
        dt = tmp.iloc[ tmp.index.map(lambda x: x in tr_list), : ]
        dt = dt.set_axis( dt.columns+offset, axis='columns', copy=False )
        dt = dt[ dt < 1000 ]
        dts_cds = {};dts_5utr={};dts_3utr={}
        for tr in tr_list:
            if tr not in dt.index:
                continue
            len_5utr = ref.annot.annot_dict[tr]['start']
            len_cds = ref.annot.annot_dict[tr]['cds_len']
            len_3utr = ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['stop'] - 3
            len_tr = ref.annot.annot_dict[tr]['cdna_len']
            if len_5utr > 0:
                cols = np.array(range(-len_5utr,0))
                dts_5utr[tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                    dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                    fill_value=0
                )
            else:
                dts_5utr[tr] = []

            cols = np.array(range(0,len_cds))
            dts_cds[tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                fill_value=0
            )
            # dts_5utr[tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
            # dts_cds[tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
            if len_3utr <= 0:
                dts_3utr[tr] = []
            else:
                cols = np.array(range(len_cds,len_tr-len_5utr))
                dts_3utr[tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                    dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                    fill_value=0
                )

    outfile_name = save_dir / f'indiv_plot_{fname}.pdf'
    pdf = PdfPages(outfile_name)
    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        
        fig, axs = plt.subplots(2,1,figsize=(10,6),sharex=True)
        tr_name = ref.id.dict_name[tr]['symbol']
        fig.suptitle(f'{tr_name}_{tr}\n{tr_list_names[ii]}')
        if mode == 'coverage':
            lw_tmp = 1000/ref.annot.annot_dict[tr]['cdna_len']
        elif mode == 'asite':
            lw_tmp = 1
        else:
            lw_tmp = 1
        color = "#818589"
        xs = {};ys = {}
        if tr not in dts_5utr:
            continue
        ys = np.hstack((dts_5utr[tr],dts_cds[tr],dts_3utr[tr]))
        xs = np.arange( -len(dts_5utr[tr]),len(dts_3utr[tr])+len(dts_cds[tr]), 1, dtype=int )
        for i in [0,1]:
            axs[i].set_ylabel('Density')
            axs[i].axvspan(0,len(dts_cds[tr]),color="#FFF8DC",alpha=0.5)
            if tr == 'ENST00000674920':#ATF4:
                # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3
                # uORF1 
                axs[i].axvspan(87-len(dts_5utr[tr]),98-len(dts_5utr[tr]),color="#7393B3",alpha=0.5)
                axs[i].axvspan(186-len(dts_5utr[tr]),365-len(dts_5utr[tr]),color="#FF69B4",alpha=0.5)

            axs[i].vlines(xs,0,ys,colors=color,lw=lw_tmp)

            if i == 0:
                axs[i].set_title(p)
                if type(pos_plot) is dict:
                    for v in pos_plot[tr]:
                        axs[0].vlines(v+len(dts_cds[tr]),0,axs[0].get_ylim()[-1],color='#FF0000',lw=lw_tmp*2)
                        axs[0].text(
                            v+len(dts_cds[tr]),
                            axs[0].get_ylim()[-1],
                            f'stop codon \n(+{v} nt)',
                            horizontalalignment='left',
                            verticalalignment='top',
                            color='#FF0000'
                            )
            elif i == 1:
                ylim_now = np.max(dts_3utr[tr])
                axs[i].set_ylim(-ylim_now*0.1,ylim_now*1.1)
                axs[i].set_xlabel('Position along the transcript (nt)') 
                    
        fig.tight_layout()
        plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close()
    pdf.close()

def indiv_plot_highlight(
    save_dir,
    load_dir,
    tr_list,
    fname,
    mode,
    ref,
    smpls,
    highlight,
    tr_list_names=[],
    ylim=[]
):
    save_dir = save_dir / f'indiv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    num_smpl = len(smpls)
    dts_cds = {};dts_5utr={};dts_3utr={}
    for p in smpls:
        if mode == 'coverage':
            dts_cds[p] = my._myload(load_dir / f'norm_cvgs_cds_{p}.joblib')
            dts_5utr[p] = my._myload(load_dir / f'norm_cvgs_5utr_{p}.joblib')
            dts_3utr[p] = my._myload(load_dir / f'norm_cvgs_3utr_{p}.joblib')
        elif 'offset' in mode:
            offset = int(mode.split('_')[-1])
            if load_dir.stem == 'norm_ribo_density':

                obj = mybin.myBinRiboNorm(
                    smpl=p,
                    sp=ref.sp,
                    save_dir=load_dir,
                    mode='start',
                    read_len_list=[],
                    is_length=False,
                    is_norm=True,
                    dirname=''
                )
                obj.decode()
                tmp = pd.DataFrame(obj.count['count5'].T.todense(),index=obj.tr['tr5'],columns=obj.pos['pos5'])
                dt = tmp.iloc[ tmp.index.map(lambda x: x in tr_list), : ]
                dt = dt.set_axis( dt.columns+offset, axis='columns', copy=False )
                # dt = dt[ dt < 1000 ]
                dt[ dt > 1000 ] = 1000
                dts_cds[p] = {};dts_5utr[p]={};dts_3utr[p]={}
                for tr in tr_list:
                    if tr not in dt.index:
                        continue
                    len_5utr = ref.annot.annot_dict[tr]['start']
                    len_cds = ref.annot.annot_dict[tr]['cds_len']
                    len_3utr = ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['stop'] - 3
                    len_tr = ref.annot.annot_dict[tr]['cdna_len']
                    if len_5utr > 0:
                        cols = np.array(range(-len_5utr,0))
                        dts_5utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                            dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                            fill_value=0
                        )
                    else:
                        dts_5utr[p][tr] = []

                    cols = np.array(range(0,len_cds))
                    dts_cds[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                        fill_value=0
                    )
                    # dts_5utr[tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
                    # dts_cds[tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
                    if len_3utr <= 0:
                        dts_3utr[p][tr] = []
                    else:
                        cols = np.array(range(len_cds,len_tr-len_5utr))
                        dts_3utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                            dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                            fill_value=0
                        )

            elif load_dir.stem == 'prep_data':
                obj = mybin.myBinRibo(
                    data_dir=ref.data_dir,
                    smpl=p,
                    sp=ref.sp,
                    save_dir=load_dir
                )
                obj.decode()
                df_data,dict_tr = obj.make_df(tr_cnt_thres=-1)
                
                dts_cds[p] = {};dts_5utr[p]={};dts_3utr[p]={}
                for tr in tr_list:
                    if tr not in dict_tr.keys():
                        continue
                    dt = dict_tr[tr]['dist5_start'].value_counts()
                    dt = dt.set_axis( dt.index+offset, axis='index', copy=False )
                    len_5utr = ref.annot.annot_dict[tr]['start']
                    len_cds = ref.annot.annot_dict[tr]['cds_len']
                    len_3utr = ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['stop'] - 3
                    len_tr = ref.annot.annot_dict[tr]['cdna_len']
                    if len_5utr > 0:
                        cols = np.array(range(-len_5utr,0))
                        dts_5utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                            dt.iloc[ (dt.index >= cols[0])*(dt.index < cols[-1]) ],
                            fill_value=0
                        )
                    else:
                        dts_5utr[p][tr] = []

                    cols = np.array(range(0,len_cds))
                    dts_cds[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.iloc[ (dt.index >= cols[0])*(dt.index < cols[-1]) ],
                        fill_value=0
                    )
                    # dts_5utr[tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
                    # dts_cds[tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
                    if len_3utr <= 0:
                        dts_3utr[p][tr] = []
                    else:
                        cols = np.array(range(len_cds,len_tr-len_5utr))
                        dts_3utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                            dt.iloc[ (dt.index >= cols[0])*(dt.index < cols[-1]) ],
                            fill_value=0
                        )

    outfile_name = save_dir / f'indiv_plot_highlight_{fname}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        fig, ax = plt.subplots(num_smpl,1,figsize=(10,1.5*num_smpl),sharex=True)
        ax[-1].set_xlabel('Position along the transcript (nt)') 
        if load_dir.stem == 'norm_ribo_density':
            ax[-1].set_ylabel('Normalized density')
        elif load_dir.stem == 'prep_data':
            ax[-1].set_ylabel('Read count')
        tr_name = ref.id.dict_name[tr]['symbol']
        if len(tr_list_names)>0:
            fig.suptitle(f'{tr_name}_{tr}\n{tr_list_names[ii]}')
        else:
            fig.suptitle(f'{tr_name}_{tr}')
        
        if mode == 'coverage':
            lw_tmp = 1000/ref.annot.annot_dict[tr]['cdna_len']
        elif mode == 'asite':
            lw_tmp = 1
        else:
            lw_tmp = 1
        color = "#818589"
        xs = {};ys = {}
        for iii,p in enumerate(smpls):
            if tr not in dts_5utr[p]:
                continue
            ys[p] = np.hstack((dts_5utr[p][tr],dts_cds[p][tr],dts_3utr[p][tr]))
            xs[p] = np.arange( -len(dts_5utr[p][tr]),len(dts_3utr[p][tr])+len(dts_cds[p][tr]), 1, dtype=int )
            ax[iii].axvspan(0,len(dts_cds[p][tr]),color="#FFF8DC",alpha=0.5)
            if tr == 'ENST00000674920':#ATF4:
                # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3
                # uORF1 
                ax[iii].axvspan(87-len(dts_5utr[p][tr]),98-len(dts_5utr[p][tr]),color="#7393B3",alpha=0.5)
                ax[iii].axvspan(186-len(dts_5utr[p][tr]),365-len(dts_5utr[p][tr]),color="#FF69B4",alpha=0.5)
            
            highlight_now = highlight[tr]
            if highlight_now[0] == 'region':
                for h in highlight_now[1:]:
                    ax[iii].axvspan(h[0],h[1],color=h[2],alpha=0.5)
            elif highlight_now[0] == 'position':
                for h in highlight_now[1]:
                    ax[iii].vlines( h[0], 0, h[1], color=h[2], lw=lw_tmp )
            
            ax[iii].vlines(xs[p],0,ys[p],colors=color,lw=lw_tmp)
            ax[iii].set_title(p)
        
        if len(ylim)>0:
            ax[0].set_ylim(ylim[0],ylim[1])
        else:
            ylim_now = np.max([ a.get_ylim()[1] for a in ax])
            ax[0].set_ylim(ax[0].get_ylim()[0],ylim_now)
        
        seq = my._seq2codon(ref.ref_cdna[tr][
            ref.annot.annot_dict[tr]["start"] : ref.annot.annot_dict[tr]["stop"]+3
        ])
        pos_codons = np.where(np.sum([seq == c for c in highlight_now[1]],axis=0))[0] * 3  
        for iii,p in enumerate(smpls):
            if highlight_now[0] == 'codon':
                for p in pos_codons:
                    ax[iii].vlines( p, 0, ylim_now, color="#880000", alpha=0.5, lw=lw_tmp )

                    
        fig.tight_layout()
        plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close()
    pdf.close()


def indiv_plot_antisense(
    save_dir,
    load_dir,
    ref,
    pairs,
    tr,
    fname
):
    save_dir = save_dir / 'indiv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)
    threshold_tr_count = -1

    if ref.sp in ['hsa','mmu']:
        tr_id = tr
        tr = ref.id.dict_name[tr_id]['symbol']
    len_cds = ref.annot.annot_dict[tr_id]['cds_len']
    len_3utr = ref.annot.annot_dict[tr_id]['start']
    len_5utr = ref.annot.annot_dict[tr_id]['cdna_len'] - len_3utr - len_cds

    outfile_name = save_dir / f'indiv_plot_{fname}.pdf'
    pdf = PdfPages(outfile_name)
    fig, axs = plt.subplots(len(pairs),1,figsize=(5,len(pairs)+1),sharex=True,)

    for i,pair in enumerate(pairs):
        dats = []
        for s in pair:
            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_cds=True)
            data = dict_tr[tr_id]['dist5_start'].value_counts() if tr_id in dict_tr.keys() else []
            dats.append(data)

        axs[i].axvspan(0,len_cds,color="#FFF8DC",alpha=0.5)
        for strand in [0,1]:
            if len(dats[strand]) == 0:
                continue
            if strand == 0:
                axs[i].vlines(
                    dats[strand].index,
                    -dats[strand].values,
                    0,
                    colors='#3912C2',
                    lw=1,
                    label='antisense'
                )
            else:
                axs[i].vlines(
                    dats[strand].index,
                    0,
                    dats[strand].values,
                    colors='#B30A0A',
                    lw=1,
                    label='sense'
                )
        # axs[i].legend()
        axs[i].set_xlim(-len_3utr,len_cds + len_5utr)
        axs[i].hlines(
            0,
            -len_3utr,
            len_cds+len_5utr,
            colors="#000000",
            lw=1)
        axs[i].set_ylabel('Counts')
        axs[i].set_title(s)
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(f'{tr_id} ({tr})')
        
        fig.tight_layout()
        # plt.show()
    axs[i].set_xlabel('Position along the transcript (nt)') 
    fig.savefig(pdf, format='pdf')
    plt.close()
    pdf.close()

def indiv_plot_subtract(
    save_dir,
    load_dir,
    tr_list,
    fname,
    mode,
    ref,
    pair
):
    mplstyle.use('fast')

    save_dir = save_dir / f'indiv_plot_subtract'
    if not save_dir.exists():
        save_dir.mkdir()
    
    dts = {};pmes = {};ifrs = {};n_reads_total = {}
    for p in pair:
        if 'offset' in mode:
            offset = int(mode.split('_')[-1])
            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=p,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            df_data,dict_tr = obj.make_df(tr_cnt_thres=16)
            
            dts[p] = {};pmes[p] = {};ifrs[p] = {}
            n_reads_total[p] = 0

            for i,tr in enumerate(tr_list):
                print(f'\r{i}/{len(tr_list)} transcripts...',end='')
                data = dict_tr[tr]['cut5'].value_counts() if tr in dict_tr.keys() else []
                if len(data) == 0:
                    continue
                data = data[ data < 1000 ]
                pos = np.array(list(data.index)) + offset
                cnts = data.values
                counts = np.zeros(ref.annot.annot_dict[tr]['cdna_len'],dtype=int)
                counts[ pos[pos<len(counts)] ] = cnts[pos<len(counts)]
                counts_cds = counts[ref.annot.annot_dict[tr]['start']:ref.annot.annot_dict[tr]['start']+ref.annot.annot_dict[tr]['cds_len']]

                if np.sum(counts_cds) < 16:
                    continue
                
                pme = my._pme(reads=counts_cds)
                ifr = np.sum(counts_cds[0::3]) / np.sum(counts_cds)

                dts[p][tr] = counts
                pmes[p][tr] = pme
                ifrs[p][tr] = ifr

                n_reads_total[p] += np.sum(counts)
    
    with gzip.open(save_dir / f'ifr_pme_{fname}.tsv.gz', 'wt') as f:
        f.write(f'transcript_id\tIFR_{pair[0]}\tIFR_{pair[1]}\tPME_{pair[0]}\tPME_{pair[1]}\n')
        for tr in tr_list:
            pme1 = pmes[pair[0]].get(tr,np.nan)
            pme2 = pmes[pair[1]].get(tr,np.nan)
            ifr1 = ifrs[pair[0]].get(tr,np.nan)
            ifr2 = ifrs[pair[1]].get(tr,np.nan)
            f.write(f'{tr}\t{ifr1}\t{ifr2}\t{pme1}\t{pme2}\n')
        

    outfile_name = save_dir / f'indiv_plot_{fname}.pdf'
    pdf = PdfPages(outfile_name)

    lw_tmp = 1
    color = "#818589"

    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        if (tr not in dts[pair[0]]) or (tr not in dts[pair[1]]):
            continue

        # figure setting
        fig, axs = plt.subplots(3,1,figsize=(10,4.5),sharex=True,sharey=True)
        axs[-1].set_xlabel('Position along the transcript (nt)') 
        axs[-1].set_ylabel('RPM')

        tr_name = ref.id.dict_name[tr]['symbol']
        fig.suptitle(f'{tr_name}_{tr}')

        titles = [pair[0],pair[1],'subtraction']
        for ax,title in zip(axs,titles):
            ax.axvspan(
                ref.annot.annot_dict[tr]['start'],
                ref.annot.annot_dict[tr]['start']+ref.annot.annot_dict[tr]['cds_len'],
                color="#FFF8DC",alpha=0.5)
            ax.set_title(title)
        
        xs = {};ys = {}
        for iii,p in enumerate(pair):
            if tr not in dts[p]:
                continue
            ys[p] = dts[p][tr] / n_reads_total[p] * 1e+6
            xs[p] = np.arange(0,ref.annot.annot_dict[tr]['cdna_len'])
            axs[iii].vlines(xs[p],0,ys[p],colors=color,lw=lw_tmp)
            
        # subtraction
        iii += 1
        ys['sub'] = ys[pair[0]] - ys[pair[1]]
        xs['sub'] = xs[pair[0]]
        axs[iii].vlines(xs['sub'][ ys['sub']>0 ],0,ys['sub'][ ys['sub']>0 ],colors="#FF0000",lw=lw_tmp)
        axs[iii].vlines(xs['sub'][ ys['sub']<0 ],0,ys['sub'][ ys['sub']<0 ],colors="#0000FF",lw=lw_tmp)
        
        ylim_now = np.max([ a.get_ylim()[1] for a in axs])
        axs[0].set_ylim(axs[0].get_ylim()[0],ylim_now)
                    
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.clf()
        plt.close()
    pdf.close()
