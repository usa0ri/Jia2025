
import pandas as pd
import pysam
import numpy as np
import re
import openpyxl
from tqdm import tqdm
from openpyxl.styles import numbers,PatternFill
import string
from pathlib import Path, PosixPath
from scipy.sparse import coo_matrix
from collections import defaultdict

import myRiboSeq.myUtil as my
import myRiboSeq.myRef as myref
import myRiboSeq.myRiboBin as mybin

# thresholds for total counts of transcripts
thresholds = [np.inf,64,32,16,8,0]

dict_pattern = {
    'hsa':'ENST([0-9]{11})',
    'mmu':'ENSMUST([0-9]{11})',
    'gga':'ENSGALT([0-9]{11})',
    'sc':'YAL([0-9A-Z-])',
    'ec':'([0-9A-Za-z-]+)'
}
dict_format = {
    'hsa':'ENST{:0>11}',
    'mmu':'ENSMUST{:0>11}',
    'gga':'ENSGALT{:0>11}',
    'sc':'YAL{%s}',
    'ec':'{%s}'
}

dict_nt_bin = {'A': 0, 'G': 1, 'C': 2, 'T': 3, 'N': 4}
nt_list = ['A', 'G', 'C', 'T', 'N']

biotype_list = ['protein-coding','ncrna','genome','pseudogene']

'''preparation of data for the downstream analyses'''
def prep_data(
    save_dir:PosixPath,
    ref_dir:PosixPath,
    data_dir:PosixPath,
    sp:str,
    fname_str = 'uniq_STAR_align_%s.bam',
    is_return_ref = False
    ):

    ref = myref.Ref(
        data_dir=data_dir,
        ref_dir=ref_dir,
        sp=sp)

    if is_return_ref:
        return ref
        
    save_dir = save_dir / 'prep_data'
    if not save_dir.exists():
        save_dir.mkdir()

    n_reads_list = [];n_reads_full_list = []
    for a in ref.exp_metadata.df_metadata['align_files']:
        smpl_name = ref.exp_metadata.df_metadata.query(f'align_files == "{a}"').sample_name.iloc[-1]
        print(f'preparing data for {smpl_name}...')
        infile = pysam.AlignmentFile(ref.data_dir /  (fname_str % a), 'rb')

        obj = mybin.myBinRibo(
            data_dir=data_dir,
            smpl=smpl_name,
            sp=sp,
            save_dir=save_dir
        )
        obj.encode(infile,ref)
        n_reads_list.append(obj.n_reads)
        obj.decode()
        df,_ = obj.make_df(tr_cnt_thres=-1,is_frame=False,is_cds=False,is_seq=False)
        idx_full = df['length'] == df['read_length']
        n_reads_full_list.append(np.sum(idx_full))
        print(f"total reads: {obj.n_reads}...")
        print(f"fully aligned reads: {np.sum(idx_full)}...")

    pd.DataFrame({
        'total_reads':n_reads_list,
        'fully_aligned_reads':n_reads_full_list
    },index=ref.exp_metadata.df_metadata['sample_name']).\
        to_csv(save_dir / 'summary.csv.gz')

    return ref

