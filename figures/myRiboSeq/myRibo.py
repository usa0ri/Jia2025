
from pathlib import Path
import joblib
import pickle
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, pearsonr

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

threshold_tr_count = 32
threshold_pos_count = 1000

def _myload(fpath):
    if fpath.suffix == '.joblib':
        with open(fpath,'rb') as f:
            out = joblib.load(f)
    elif fpath.suffix == '.pkl':
        with open(fpath,'rb') as f:
            out = pickle.load(f)
    elif fpath.suffix == '.gz':
        out = pd.read_csv(fpath,index_col=0,header=0)
    print(f'{fpath} loaded.')
    return out

def _mysave(fpath,obj):
    if fpath.suffix == '.joblib':
        with open(fpath,'wb') as f:
            joblib.dump(obj,f,compress=3)
    elif fpath.suffix == '.pkl':
        with open(fpath,'wb') as f:
            pickle.dump(obj,f)
    elif '.csv.gz' in str(fpath):
        obj.to_csv(fpath,compression='gzip')
    print(f'{fpath} saved.')

def _load_dict_tr(
        load_dir,
        s,
        threshold_tr_count,
        biotype=''
    ):
    thres_now = thresholds[1: np.where([threshold_tr_count == x for x in thresholds])[0][0]+1 ]
    dict_tr = {}
    label_tr = {}
    idx_tmp = 0
    for thres in thres_now:
        if biotype in ['','tr']:
            fname = f'dict_tr_{s}_tr{thres}.joblib'
        elif biotype == 'ncrna':
            fname = f'dict_ncrna_{s}_tr{thres}.joblib'
        with open(load_dir / fname, "rb") as f:
            dict_tr_now = joblib.load(f)
            dict_tr |= dict_tr_now
            label_tr[thres] = (idx_tmp,idx_tmp+len(dict_tr_now))
            idx_tmp += len(dict_tr_now)
    print(f'dict_tr of {s} >={threshold_tr_count} has been loaded.')
    return dict_tr,label_tr
    
def _calc_norm_ribo_density(
    df_data,
    dict_tr,
    label_tr,
    read_len,
    mode):
    # ribosomde density from start codon
    # read count: [position] * [transcript]
    # df_data = df_data.query('(read_length >= 15) and (read_length < 35)')
    pkl_out = {
        'count5':[],
        'tr5':[],
        'pos5':[],
        'count3':[],
        'tr3':[],
        'pos3':[],
        'label_tr':label_tr
        }
    for end in [5,3]:
        if mode in ('start','stop'):
            count5_pos = df_data.query((
                f'(dist{end}_start > -600) and '
                f'(dist{end}_stop < 600) and '
                f'(read_length >= {read_len[0]}) and '
                f'(read_length <= {read_len[1]})'))[["tr_id",f"dist{end}_{mode}"]]\
                .pivot_table(index=f"dist{end}_{mode}",columns="tr_id",aggfunc=len,fill_value=0)
        elif mode == '3end':
            tr_len_list = np.array([
                dict_tr[tr]["tr_info"]["cdna_len"]
                for tr in df_data["tr_id"]
            ])
            df_data[f'dist{end}_start2'] = df_data[f'dist{end}_start'] + df_data['start'] - tr_len_list
            count5_pos = df_data.query((
                f'(read_length >= {read_len[0]}) and '
                f'(read_length <= {read_len[1]})'))[["tr_id",f"dist{end}_start2"]]\
                .pivot_table(index=f"dist{end}_start2",columns="tr_id",aggfunc=len,fill_value=0)
            
        if len(count5_pos) > 0:
            # normalize for each gene -> normalize for each position
            tr_norm_term = np.array([
                len(dict_tr[tr]["df"]) / dict_tr[tr]["tr_info"]["cds_len"]
                for tr in count5_pos.columns
            ])
            count5_pos_norm = count5_pos.values / tr_norm_term[None,:]
            count5_pos_norm_fill = np.zeros((np.max(count5_pos.index)-np.min(count5_pos.index)+1, len(count5_pos.columns) ))
            count5_pos_norm_fill[ np.array(count5_pos.index)-np.min(count5_pos.index), : ] = count5_pos_norm
            # for i,tr in enumerate(count5_pos.columns):
            #     count5_pos_norm_fill[ np.array(count5_pos.index)-np.min(count5_pos.index), i ] += count5_pos_norm[:,i]
            coo = coo_matrix(count5_pos_norm_fill,shape=count5_pos_norm_fill.shape)
            dict_col = np.array(list(range( np.min(count5_pos.index), np.max(count5_pos.index)+1 )))
            dict_row = list(count5_pos.columns)
            pkl_out[f'count{end}'] = coo
            pkl_out[f'tr{end}'] = dict_row
            pkl_out[f'pos{end}'] = dict_col
            
    return pkl_out

def calc_norm_ribo_density(
    save_dir,
    load_dir,
    smpls,
    read_len_list=[],
    is_length=False
):
    save_dir = save_dir / f'norm_ribo_density'
    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)

    threshold_tr_count = 0
    for s in smpls:
        print(f'\ncalculating normalized ribosome density for {s}...')
        df_data = _myload(load_dir / f'df_{s}.joblib')
        dict_tr,label_tr = _load_dict_tr(load_dir,s,threshold_tr_count)
        df_data = df_data[[x in dict_tr.keys() for x in df_data['tr_id']]]

        df_data_readlen = df_data.groupby('read_length')
        read_len_list_all = list(df_data_readlen.groups.keys())
        
        for mode in ['start','stop']:
            # all read length
            print(f'all read length...')
            pkl_out = _calc_norm_ribo_density(
                df_data,
                dict_tr,
                label_tr,
                [read_len_list_all[0],read_len_list_all[-1]],
                mode)
            _mysave(save_dir / f'df_norm_density_{mode}_{s}.pkl',pkl_out)

            if len(read_len_list)>0:
                print(f'read length {read_len_list[0]} to {read_len_list[1]}...')
                pkl_out = _calc_norm_ribo_density(
                    df_data,
                    dict_tr,
                    label_tr,
                    [read_len_list[0],read_len_list[-1]],
                    mode)
                _mysave(save_dir / f'df_norm_density_{mode}_{s}_{read_len_list[0]}-{read_len_list[-1]}.pkl',pkl_out)


            # for each read length
            if is_length:
                pkl_outs = {}
                for read_len in read_len_list_all:
                    print(f'\r{read_len} read length...',end='')
                    pkl_outs[read_len] = _calc_norm_ribo_density(
                        df_data_readlen.get_group(read_len),
                        dict_tr,
                        label_tr,
                        [read_len,read_len],
                        mode)
                _mysave(save_dir / f'df_norm_density_{mode}_readlen_{s}.joblib',pkl_outs)

def _agg_ribo(
    load_dir,
    s,
    xlim_now,
    read_len_list
    ):
    xs = {}
    ys = {}
    for base in ('start','stop'):
        if len(read_len_list)>0:
            dt = _myload(load_dir / f'df_norm_density_{base}_{s}_{read_len_list[0]}-{read_len_list[1]}.pkl')
        else:
            dt = _myload(load_dir / f'df_norm_density_{base}_{s}.pkl')
        # filter transcripts > threshold_tr_count
        if type(threshold_tr_count) is int:
            idx_columns = np.array(dt['count5'].sum(axis=0)[0] > threshold_tr_count).flatten()
        elif type(threshold_tr_count) is dict:
            tr_list = threshold_tr_count['tr_list']
            idx_columns = [np.where(np.array(dt['tr5']) == tr)[0][0] for tr in tr_list]
        count = dt['count5'].tocsc()[:,np.array(idx_columns)].tolil()
        num_nonzero = count.nnz
        # filter positions < threshold_pos_count
        count[ count > threshold_pos_count ] = 0
        print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

        # FIXME
        idx_ = np.where(
            (np.array(dt['pos5'])>=xlim_now[base][0]) * (np.array(dt['pos5'])<=xlim_now[base][1]))[0]
        x = np.array(dt['pos5'])[idx_]
        y = np.array(count.mean(axis=1)).flatten()[idx_]
        # idx_pos = [np.where(np.array(dt['pos5']) == x)[0][0] for x in xlim_now[base]]
        # x = np.array(list(range(xlim_now[base][0],xlim_now[base][1])))
        # y = np.array(count.mean(axis=1)).flatten()[idx_pos[0]:idx_pos[1]]
        xs[base] = x
        ys[base] = y
    return xs,ys

'''aggregation plot (metagene plot)
    relative distance from start/stop site (x axis) v.s. mean read density (y axis)
'''
def aggregation_ribo(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    threshold_tr_count,
    threshold_pos_count,
    col_list,
    labels,
    fname,
    read_len_list=[]
    ):
    
    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)
    outfile_name = save_dir / fname

    xlim_now = {'start':plot_range5,'stop':plot_range3}
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    flag_legend = True
    fig,axs = plt.subplots(len(col_list),2,figsize=(4.5,4.5))
    for n,s in enumerate(col_list.keys()):
        print(f'\naggregation plots for {s}...')
        
        xs,ys = _agg_ribo(
            load_dir,
            s,
            xlim_now,
            threshold_tr_count,
            threshold_pos_count,
            read_len_list)

        # frame
        frames = [''] * 3
        for i in range(3):
            for j,base in enumerate(["start","stop"]):
                idx = (xs[base] % 3 == i)
                frames[i], = axs[n,j].plot(
                    xs[base][idx]+15,
                    ys[base][idx],
                    color=color_frame[f'frame{i}'],
                    label=f'Frame {i}',
                    linewidth=2)
                if n==len(col_list.keys())-1:
                    axs[n,j].set_xlabel(base.capitalize(),fontsize=12)
                axs[n,j].set_ylabel("Mean reads",fontsize=12)
                if j==0:
                    axs[n,j].spines['right'].set_visible(False)
                else:
                    axs[n,j].spines['left'].set_visible(False)
                    axs[n,j].get_yaxis().set_visible(False)
                axs[n,j].set_xlim(xlim_now[base][0],xlim_now[base][1]+30)
                if j==0:
                    axs[n,j].set_xticks([0,150,300])
                else:
                    axs[n,j].set_xticks([-300,-150,0])
                axs[n,j].set_xticklabels(labels=axs[n,j].get_xticklabels(),fontsize=12)
        if flag_legend:
            axs[n,0].legend(
                handles=[frames[0],frames[1],frames[2]],
                frameon=False,
                fontsize=10,
                loc='upper right',
                labelspacing = 0.25)
            flag_legend = False

    # match y axis
    ylim_max = np.max([np.max([axs[n,0].get_ylim(),axs[n,1].get_ylim()]) for n in range(len(col_list.keys()))])
    for n in range(len(col_list.keys())):
        axs[n,0].set_ylim(-ylim_max*0.05,ylim_max*1.05)
        axs[n,1].set_ylim(-ylim_max*0.05,ylim_max*1.05)
        for j in range(2):
            axs[n,j].set_yticklabels(labels=axs[n,j].get_yticklabels(),fontsize=12)
        axs[n,1].text(
            x=-30,
            y=ylim_max*0.9,
            s=labels[n],
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='top',
        )
    axs[n,0].text(
            x=-100,
            y=-4,
            s='Distance (nt)',
            fontsize=12,
            horizontalalignment='left',
            verticalalignment='top',
        )
        
        
    fig.tight_layout()
    plt.show()
    # fig.savefig(outfile_name.with_suffix('.svg'))
    fig.savefig(pdf, format='pdf')
    plt.close('all')

    pdf.close()

def ATF4_plot(
    save_dir,
    load_dir,
    fname,
    col_list,
    labels,
    ref
):
    offset = 15
    if 'Homo_sapiens' in ref.ref_dir.stem:
        tr_list = ['ENST00000674920']
        tr_list_names = ['ATF4']
    elif 'Mus_musculus' in ref.ref_dir.stem:
        tr_list = ['ENSMUST00000109605']
        tr_list_names = ['Atf4']

    num_smpl = len(col_list)
    dts_cds = {};dts_5utr={};dts_3utr={}
    for p in col_list.keys():
        dt = _myload(load_dir / f'df_norm_density_start_{p}.pkl')
        tmp = pd.DataFrame(dt['count5'].T.todense(),index=dt['tr5'],columns=dt['pos5'])
        dt = tmp.iloc[ tmp.index.map(lambda x: x in tr_list), : ]
        dt = dt.set_axis( dt.columns+offset, axis='columns', copy=False )
        dt = dt[ dt < 1000 ]
        dts_cds[p] = {};dts_5utr[p]={};dts_3utr[p]={}
        for tr in tr_list:
            if tr not in dt.index:
                continue
            len_5utr = np.min([ref.annot.annot_dict[tr]['start'],599-16])
            len_cds = np.min([ref.annot.annot_dict[tr]['cds_len'],dt.columns[-1]])
            len_tr = ref.annot.annot_dict[tr]['cdna_len']
            dts_5utr[p][tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
            dts_cds[p][tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
            if len_cds == dt.columns[-1]:
                dts_3utr[p][tr] = []
            else:
                dts_3utr[p][tr] = dt.loc[tr, np.array(range(len_cds,len_tr-len_5utr)) ]

    outfile_name = save_dir / fname
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        fig, ax = plt.subplots(num_smpl,1,figsize=(6,4.5),sharex=True,sharey=True)
        lw_tmp = 2
        color = "#818589"
        xs = {};ys = {}
        for iii,p in enumerate(col_list.keys()):
            if tr not in dts_5utr[p]:
                continue
            ys[p] = np.hstack((dts_5utr[p][tr],dts_cds[p][tr],dts_3utr[p][tr]))
            # xs[p] = np.arange( -len(dts_5utr[p][tr]),len(dts_3utr[p][tr])+len(dts_cds[p][tr]), 1, dtype=int )
            xs[p] = np.arange( 0, len(dts_5utr[p][tr])+len(dts_3utr[p][tr])+len(dts_cds[p][tr]), 1, dtype=int )
            ax[iii].axvspan(len(dts_5utr[p][tr]),len(dts_5utr[p][tr])+len(dts_cds[p][tr]),color="#FFF8DC",alpha=0.5)
            if tr == 'ENST00000674920':#ATF4:
                # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3
                # uORF1 
                ax[iii].axvspan(87,98,color="#7393B3",alpha=0.5)
                ax[iii].axvspan(186,365,color="#FF69B4",alpha=0.5)
            
            ax[iii].vlines(xs[p],0,ys[p],colors=color,lw=lw_tmp)
            ax[iii].spines['right'].set_visible(False)
            ax[iii].spines['top'].set_visible(False)
            ax[iii].text(
                x=1400,
                y=90,
                s=labels[iii],
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='top',
                )
        
        ylim_now = np.max([ a.get_ylim()[1] for a in ax])
        xticks_now = np.arange(0,1401,200)
        yticks_now = np.arange(0,101,50)
        for a in ax:
            a.set_ylim(a.get_ylim()[0],ylim_now)
            a.set_yticks(yticks_now)
            a.set_yticklabels(labels=yticks_now,fontsize=12)
            a.set_xticks(xticks_now)
            a.set_xticklabels(labels=xticks_now,fontsize=12)
        fig.tight_layout()
        plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close()
    pdf.close()

def ratio_5utr_cds(
    save_dir,
    load_dir,
    pairs,
    labels,
    ref
):
    xs = {};ys={};zs={}
    x_atf4 = {};y_atf4 = {};z_atf4 = {}
    for pair,label in zip(pairs,labels):
        fname_out = save_dir / f'df_log2ratio_5utr_cds_{pair[0]}_{pair[1]}.csv.gz'
        if fname_out.exists():
            df = pd.read_csv(fname_out,header=0,index_col=0)
            xs[label] = df[f'ratio_{pair[0]}']
            ys[label] = df[f'ratio_{pair[1]}']
            zs[label] = df['density']
            continue
        dfs = {}
        for p in pair:
            dict_tr,label_tr = _load_dict_tr(load_dir,p,threshold_tr_count)
            dat_5utr = [];dat_cds = [];trs = []
            len_tr = len(dict_tr)
            for i,(tr,v) in enumerate(dict_tr.items()):
                print(f'\r{i}/{len_tr} transcripts',end='')
                uniq,cnt_uniq = np.unique(v['df']['cds_label'].values,return_counts=True)
                if ('5UTR' not in uniq) or ('CDS' not in uniq):
                    continue
                cnt_5utr = cnt_uniq[ uniq == '5UTR' ][0]
                cnt_cds = cnt_uniq[ uniq == 'CDS' ][0]
                if cnt_cds>0 and cnt_5utr>0:
                    dat_5utr.append(cnt_5utr)
                    dat_cds.append(cnt_cds)
                    trs.append(tr)
            df_now = pd.DataFrame({
                '5UTR':dat_5utr,
                'CDS':dat_cds
                },index=trs)
            df_now.apply(lambda x: np.log2(x))
            dfs[p] = df_now
        df_out = pd.merge(dfs[pair[0]],dfs[pair[1]],how='inner',left_index=True,right_index=True,suffixes=pair)

        x = (df_out[f'5UTR{pair[0]}'] / df_out[f'CDS{pair[0]}']).apply(lambda x: np.log2(x)).values
        y = (df_out[f'5UTR{pair[1]}'] / df_out[f'CDS{pair[1]}']).apply(lambda x: np.log2(x)).values
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        if 'Homo_sapiens' in ref.ref_dir.stem:
            idx_atf4 = np.where(df_out.index == 'ENST00000674920')[0]
        elif 'Mus_musculus' in ref.ref_dir.stem:
            idx_atf4 = np.where(df_out.index == 'ENST00000674920')[0]

        idx = z.argsort()
        x_, y_, z_ = x[idx[idx_atf4]], y[idx[idx_atf4]], z[idx[idx_atf4]]
        x, y, z = x[idx], y[idx], z[idx]
        xs[label] = x
        ys[label] = y
        zs[label] = z
        x_atf4[label] = x_
        y_atf4[label] = y_
        z_atf4[label] = z_

        df_out = df_out.iloc[idx,:].copy()
        df_out[f'ratio_{pair[0]}'] = x
        df_out[f'ratio_{pair[1]}'] = y
        df_out['density'] = z
        df_out.to_csv(fname_out)

    # Pearson
    rs = {}
    for label in labels:
        rs[label] = pearsonr(xs[label],ys[label])

    yticks_now = [-15,-10,-5,0,5,10]
    xticks_now = [-15,-10,-5,0,5,10]
    outfile_name = save_dir / 'ratio_5utr_cds'   
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    fig, ax = plt.subplots(1,len(labels),figsize=(4.5,2.5),sharex=True,sharey=True)
    for i,label in enumerate(labels):
        ax[i].scatter(
            xs[label],ys[label],c=zs[label],s=1
        )
        ax[i].plot([-15,10],[-15,10],linewidth=1,color='red')
        # ax[i].scatter(
        #     x_atf4[label],y_atf4[label],c="red",s=20
        # )
        ax[i].set_xlabel(label[0],fontsize=12)
        ax[i].set_ylabel(label[1],fontsize=12)
        ax[i].set_xlim(-15,10)
        ax[i].set_ylim(-15,10)
        ax[i].text(
            x=-14,
            y=9,
            s='Pearson\'s\n$\mathit{r}$ = ' + str(round(rs[label][0],3)),
            fontsize=12,
            horizontalalignment='left',
            verticalalignment='top',)
        ax[i].set_yticks(yticks_now)
        ax[i].set_yticklabels(labels=yticks_now,fontsize=12)
        ax[i].set_xticks(xticks_now)
        ax[i].set_xticklabels(labels=xticks_now,fontsize=12)
        ax[i].set_title('Log2 (5\'UTR/CDS)',fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig(pdf, format='pdf')
    pdf.close()
