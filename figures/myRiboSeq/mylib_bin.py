from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle
import re
import gzip
import matplotlib as mpl
from Bio import SeqIO,SeqRecord

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = font_prop.get_name()

thresholds = [np.inf,64,32,16,8,0]

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


color_atcg = {
    'A':'#008000',
    'G':'#FFA500',
    'T':'#FF0000',
    'C':'#0000FF'
}


codon2aa_table = {
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
aa2codon_table = {
    'I':('ATA','ATC','ATT'),
    'M':['ATG'],
    'T':('ACA','ACC','ACG','ACT'),
    'N':('AAC','AAT',),
    'K':('AAA','AAG'),
    'S':('AGC','AGT','TCA','TCC','TCG','TCT'),
    'L':('CTA','CTC','CTG','CTT','TTA','TTG'),
    'P':('CCA','CCC','CCG','CCT'),
    'H':('CAC','CAT'),
    'Q':('CAA','CAG'),
    'R':('CGA','CGC','CGG','CGT','AGA','AGG'),
    'V':('GTA','GTC','GTG','GTT'),
    'A':('GCA','GCC','GCG','GCT'),
    'D':('GAC','GAT'),
    'E':('GAA','GAG'),
    'G':('GGA','GGC','GGG','GGT'),
    'F':('TTC','TTT'),
    'Y':('TAC','TAT'),
    '_':('TAA','TAG','TGA'),
    'C':('TGC','TGT'),
    'W':['TGG']
    }

anticodon_table = {
    'A':('AGC','TGC'),
    'R':('ACG','TCG'),
    'N':('GTT'),
    'D':('GTC'),
    'C':('GCA'),
    'Q':('CTG'),
    'E':('CTC','TTC'),
    'G':('GCC'),
    'H':('GTG'),
    'I':('GAT','AAT'),
    'L':('CAG','TAG','CAA','TAA'),
    'K':('CTT','TTT'),
    'M':('CAT'),
    'F':('GAA'),
    'P':('TGG'),
    'S':('GCT','TCT','AGT'),
    'T':('CGT','AGT'),
    'W':('CCA'),
    'Y':('GTA'),
    'V':('TAC','CAC'),
    '_':''
}

biotype_list = ['protein-coding','ncrna','pseudogene','genome']

def get_gzip_uncompressed_size(filename):
    with open(filename, 'rb') as f:
        # Seek to the end of the file
        f.seek(-4, 2)
        # Read the last 4 bytes
        size_bytes = f.read(4)
        # Unpack the 4 bytes into an integer (little-endian)
        return struct.unpack('<I', size_bytes)[0]

def _get_spaced_colors(n): 
    hsv_colors_ = [
        (x*1.0/n, 0.5, 0.9)
        for x in range(n)]
    hsv_colors = hsv_colors_[::3] + hsv_colors_[1::3] + hsv_colors_[2::3]
    if len(hsv_colors) != n:
        hsv_colors += hsv_colors_[ [i for i in range(n) if hsv_colors_[i] not in hsv_colors] ]
    rgb_colors = [
        mpl.colors.hsv_to_rgb(hsv)
        for hsv in hsv_colors
    ]
    hex_colors = [
        mpl.colors.to_hex(rgb)
        for rgb in rgb_colors
    ]
     
    return  rgb_colors,hex_colors


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def _get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


'''fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)'''
def _colorFader(c1,c2,mix=0):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def _load_fa(
    fa_file
):
    from Bio import SeqIO
    dict_fa = {}
    with gzip.open(fa_file,mode='rt') as f:
        dict_fa = SeqIO.index(f, "fasta")
    return dict_fa

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

def _getcodon(tr,ref):
    return np.array(re.split('(...)',ref.ref_cdna[tr][
            ref.annot.annot_dict[tr]["start"] : ref.annot.annot_dict[tr]["stop"]
            ])[1::2])

def _seq2codon(seq):
    return np.array(re.split('(...)',seq)[1::2])

def _seq2aa(seq):
    if type(seq) is SeqRecord.SeqRecord:
        seq = str(seq.seq)
    if 'N' in seq:
        aa = []
        for c in re.split('(...)',seq)[1::2]:
            if 'N' in c:
                aa.append('')
            else:
                aa.append(codon2aa_table[c])
        return aa
    else:
        return np.array([codon2aa_table[c] for c in re.split('(...)',seq)[1::2]])

def _rev(seq,is_rev=True):
    dict_rev = {
        'A':'T',
        'C':'G',
        'T':'A',
        'G':'C',
        'N':'N'}
    if is_rev:
        out = ''.join([ dict_rev[x] for x in list(seq[::-1])])
    else:
        out = ''.join([ dict_rev[x] for x in list(seq)])
    return out

def _pme(reads:np.array,is_norm=False):

    if is_norm:
        norm_factor = np.ceil((len(reads)/3) / np.sum(reads)).astype(int)
    else:
        norm_factor = 1
    n_codon = np.floor(len(reads)/(3*norm_factor)).astype(int)
    # FIXME
    reads = reads[:n_codon*3*norm_factor]
    codons = reads.reshape((n_codon,3*norm_factor)).sum(axis=1)
    total_reads = np.sum(codons)
    if total_reads < 10:
        return 0
    p = codons / total_reads

    h = -np.sum( p[p>0] * np.log2(p[p>0]) )
    max_h = np.log2( n_codon )
    
    return h/max_h

def _pme_ORQAS(reads:np.array):

    # counts in each codon
    new_counts_ribo = []
    for n2,pos in enumerate(reads):
        if n2 %3 == 0:
            new_counts_ribo.append(pos)
        else:
            new_counts_ribo[-1] = new_counts_ribo[-1] + pos

    l = len(new_counts_ribo)
    n = np.sum(new_counts_ribo)

    if n < 10:
        pme_t = 0
        hm = 0
    
    else:
        if n > l:
            rl = float(1)
        else:
            rl = float(round(l/n))
        c = 0
        nr = 0
        h = 0
        hm = 0
        for i,count in enumerate(reads):
            nr = nr + count
            c += 1
            if c == rl:
                p = float(nr)/float(n)
                if p > 0:
                    h = h + p*np.log2(p)
                pm = (n/(l/rl))/n
                hm = hm + (pm*np.log2(pm))
                c = 0
                nr = 0
        pme_t = h/hm
    
    return pme_t


class Ref:

    def __init__(
        self,
        data_dir,
        ref_dir,
        sp,
        annot_key='tr_id'
        ):

        self.sp = sp

        if data_dir != '':
            data_dir = Path(data_dir)
            self.data_dir = data_dir
            exp_metadata_file = data_dir / "exp_metadata.csv"
            '''parse metadata of the experiment'''
            self.exp_metadata = ExpMetadata(exp_metadata_file)
            print("experimental metadata has been loaded...")

        if ref_dir != '':
            ref_dir = Path(ref_dir)
            self.ref_dir = ref_dir
            ref_cdna_file = ref_dir / "seq_selected_transcripts.fa.gz"
            annot_file = ref_dir / "annot_selected_transcripts.gff.gz"
            id_name_file = ref_dir / "tr_info.txt.gz"
        
            '''reference transcriptome data (longest CDS in each transcript)'''
            self.ref_cdna = RefcDNA(Path(ref_cdna_file))
            print("reference cDNA has been loaded...")

            '''annotation'''
            self.annot = Annot(Path(annot_file),sp,key=annot_key)
            print("annotation has been loaded...")

            '''transcript id and name'''
            self.id = IDName(Path(id_name_file))
            print("transcript ID/name information has been loaded...")
        
        
class ExpMetadata():
    def __init__(self,exp_metadata_file):
        if exp_metadata_file is None:
            raise Exception('exp_metadata (*.csv) should be specified')
        self.original_path = Path(exp_metadata_file)
        delimiter = ","
        self.df_metadata = pd.read_csv(exp_metadata_file, header=0, delimiter=delimiter, index_col=0)
    
    def __getattr__(self,attribute):
        if attribute not in (self.df_metadata.columns):
            raise Exception(f"invalid attributes\n{attribute}")
        return self.df_metadata.query(f"column=='{attribute}'").values[0]

class RefcDNA(dict):
    def __init__(self,ref_cdna_file):
        if ref_cdna_file is None:
            raise Exception('ref_cdna_file (*.fa) should be specified')
        # self.original_path = Path(ref_cdna_file)
        if ref_cdna_file.suffix == '.fa':
            fopen = lambda x: open(x,mode='r')
        elif ref_cdna_file.suffix == '.gz':
            fopen = lambda x: gzip.open(x,mode='rt')
        self.original_path = ref_cdna_file
        self.fopen = fopen
        with self.fopen(self.original_path) as f:
            dict_seq = SeqIO.to_dict(SeqIO.parse(f, "fasta"))
        self.dict_seq = dict_seq
    
    def __getitem__(self, key):
        return str(self.dict_seq[key].seq)

    def items(self):
        v_str = [ str(v.seq) for v in self.dict_seq.values() ]
        k_str = [ k for k in self.dict_seq.keys() ]
        
        return dict(zip(k_str, v_str)).items()

    def keys(self):
        k_str = [ k for k in self.dict_seq.keys() ]
        return k_str
    
    def values(self):
        v_str = [ str(v.seq) for v in self.dict_seq.values() ]
        return v_str

    
class Annot():
    def __init__(
            self,
            annot_file,
            sp,
            mode='tr',
            key='tr_id'):
        
        if annot_file is None:
            raise Exception('annot_file should be specified')

        self.original_path = annot_file

        if mode == 'tr':
            if key == 'tr_id':
                dt = pd.read_csv(annot_file,header=None,index_col=0,usecols=[0,1,2,3,4,6,8], sep="\t")\
                    .set_axis([
                        'gene_id',# gene id
                        'chr',# chromosome
                        'start',# within transcripts
                        'stop',# within transcripts
                        'strand',
                        'attrs'
                    ],axis=1)
            elif key == 'gene_id':
                dt = pd.read_csv(annot_file,header=None,index_col=1,usecols=[0,1,2,3,4,6,8], sep="\t")\
                    .set_axis([
                        'tr_id',# transcript id
                        'chr',# chromosome
                        'start',# within transcripts
                        'stop',# within transcripts
                        'strand',
                        'attrs'
                    ],axis=1)
            tmp = dt['attrs'].apply(lambda x: x.split(';'))
            dt.drop('attrs',axis=1,inplace=True)
            dt.loc[:,'start'] = dt['start']-1
            # FIXME
            if sp in ['hsa','mmu','sc']:
                dt.loc[:,'stop'] = dt['stop']-1-3
                dt['cds_len'] = dt['stop'] - dt['start']
            else:
                dt.loc[:,'stop'] = dt['stop']-1
                dt['cds_len'] = dt['stop'] - dt['start'] + 3
            dt['cdna_len'] = [
                int(tmp.loc[tr][0][9:])
                for tr in dt.index
            ]
        elif mode == 'gene':
            dt = pd.read_csv(annot_file,header=None,index_col=0,usecols=[0,1,2,3,4,6,8],sep="\t")\
                    .set_axis([
                        'chr',# chromosome
                        'gene_biotype',
                        'start',# within chromosome
                        'stop',# within chromosome
                        'strand',
                        'attrs'
                    ],axis=1)
            tmp = dt['attrs'].apply(lambda x: x.split(';'))
            dt.drop('attrs',axis=1,inplace=True)

            tr_ids = [];regions = []
            for gene in dt.index:
                tr_ids.append(tmp.loc[gene][0])
                region = np.array([[int(y) for y in x.split('_')] for x in tmp.loc[gene][1:]])
                region[:,0] -= dt.loc[gene,'start']
                region[:,1] -= dt.loc[gene,'start']
                regions.append(region.tolist())
            dt['tr_id'] = tr_ids
            dt['regions'] = regions

            self.biotypes = list(dt['gene_biotype'].unique())

            if key == 'tr_id':
                dt = dt.reset_index().set_index('tr_id').set_axis(['gene_id','chr','gene_biotype','start','stop','strand','regions'],axis=1)
                
        self.annot_dict = dt.to_dict(orient='index')
        
    def __getattr__(self,attribute):
        return getattr(self.df_start_stop,attribute)

class IDName():
    def __init__(self,id_name_file,biotype='protein-coding'):
        if id_name_file is None:
            raise Exception('id_name_file should be specified')
        self.original_path = Path(id_name_file)

        if biotype == 'protein-coding':
            df_name = pd.read_csv(id_name_file,header=None,index_col=0,sep='\t')
            if len(df_name.columns) == 2:
                dict_name = df_name.set_axis(['symbol','other'],axis=1)\
                    .to_dict(orient='index')
            elif len(df_name.columns) == 3:
                dict_name = df_name.set_axis(['symbol','other1','other2'],axis=1)\
                    .to_dict(orient='index')
        elif (biotype == 'ncrna') or (biotype == 'pseudogene'):
            df_name = pd.read_csv(id_name_file,header=0,index_col=1,sep='\t')
            # FIXME
            # remove duplicated transcript ID
            df_name = df_name[ ~df_name.index.duplicated(keep='first') ]
            dict_name = df_name.set_axis(['gene_id','symbol'],axis=1)\
                .to_dict(orient='index')

        self.dict_name = dict_name

def _name2id(ref,tr_lists_query):
    tr_lists_ = [
        [
            k
            for k,v in ref.id.dict_name.items()
            if v['symbol'] == tr
        ]
        for tr in tr_lists_query
    ]
    tr_lists = [
        [
            tr
            for tr in tr_list
            if tr in ref.ref_cdna.keys()
        ][0]
        for tr_list in tr_lists_
    ]
    return tr_lists

class RefGenome2Transcriptome:

    def __init__(
        self,
        data_dir,
        ref_dir
        ):

        data_dir = Path(data_dir)
        ref_dir = Path(ref_dir)
        self.data_dir = data_dir
        self.ref_dir = ref_dir

        exp_metadata_file = data_dir / "exp_metadata.csv"

        '''protein-coding genes'''
        ref_cdna_file = ref_dir / "seq_transcripts.fa.gz"
        annot_file = ref_dir / 'annot_transcripts.gff.gz'
        id_name_file = ref_dir / "tr_info.txt.gz"

        '''noncoding RNAs'''
        ref_ncrna_file = ref_dir / "seq_ncrna.fa.gz"
        annot_ncrna_file = ref_dir / 'annot_ncrna.gff.gz'
        id_name_ncrna_file = ref_dir / "ncrna_info.txt.gz"
        
        ''' parse metadata of the experiment'''
        self.exp_metadata = ExpMetadata(exp_metadata_file)
        print("experimental metadata has been loaded...")
        
        '''reference transcriptome data (longest CDS in each transcript)'''
        self.ref_cdna = RefcDNA(Path(ref_cdna_file))
        self.ref_ncrna = RefcDNA(Path(ref_ncrna_file))
        print("reference cDNA and ncRNA has been loaded...")

        '''annotation'''
        self.annot = Annot(Path(annot_file))
        self.annot_ncrna = Annot(Path(annot_ncrna_file))
        print("annotation has been loaded...")

        '''transcript id and name'''
        self.id = IDName(Path(id_name_file))
        self.id_ncrna = IDName(Path(id_name_ncrna_file))
        print("transcript ID/name information has been loaded...")

class RefGenome:

    def __init__(
        self,
        data_dir,
        ref_dir,
        ref_dir_genome,
        sp,
        biotypes:list
        ):

        self.ref_dir = Path(ref_dir)
        self.ref_dir_genome = Path(ref_dir_genome)
        self.data_dir = Path(data_dir)
        self.sp = sp
        self.biotype_list = biotypes
        self.dict_biotypes = {}

        assert np.all([b in biotype_list for b in biotypes])

        self.exp_metadata = ExpMetadata(Path(data_dir) / "exp_metadata.csv")
            
        if 'ncrna' in biotypes:
            self.exp_metadata_ncrna = ExpMetadata(Path(data_dir) / "exp_metadata_ncrna.csv")
            
        if 'pseudogene' in biotypes:
            self.exp_metadata_pseudogene = ExpMetadata(Path(data_dir) / "exp_metadata_pseudogene.csv")
        
        self.ref_seq_gene = {}
    
    def get_annot_tr(self,biotype,annot_key='tr_id'):
        if biotype == 'protein-coding':
            return Annot( self.ref_dir / 'annot_selected_transcripts.gff.gz', sp=self.sp, mode='tr', key=annot_key)
        elif biotype == 'ncrna':
            return Annot( self.ref_dir_genome / 'annot_genes_ncrna.gff.gz', sp=self.sp, mode='gene',key=annot_key)
        elif biotype == 'pseudogene':
            return Annot( self.ref_dir_genome / 'annot_genes_pseudogene.gff.gz', sp=self.sp, mode='gene',key=annot_key )
    
    def get_annot_gene(self,biotype,key):
        if biotype == 'protein-coding':
            return Annot( self.ref_dir_genome / 'annot_genes.gff.gz', sp=self.sp, mode='gene', key=key)
        elif biotype == 'ncrna':
            return Annot( self.ref_dir_genome / 'annot_genes_ncrna.gff.gz', sp=self.sp, mode='gene',key=key)
        elif biotype == 'pseudogene':
            return Annot( self.ref_dir_genome / 'annot_genes_pseudogene.gff.gz', sp=self.sp, mode='gene',key=key)
    
    def get_seq_tr(self,biotype):
        if biotype == 'protein-coding':
            return RefcDNA(Path(self.ref_dir / 'seq_selected_transcripts.fa.gz'))
        elif biotype == 'ncrna':
            return RefcDNA(Path(self.ref_dir_genome / 'ref_ncrna_seq.fa.gz'))
        elif biotype == 'pseudogene':
            return RefcDNA(Path(self.ref_dir_genome / 'ref_pseudogene_seq.fa.gz'))
    
    def get_seq_gene(self):
        return RefcDNA(Path(self.ref_dir_genome / 'seq_genes.fa.gz'))

    def get_id_tr(self,biotype):
        if biotype == 'protein-coding':
            return IDName(self.ref_dir / "tr_info.txt.gz")
        elif biotype == 'ncrna':
            return IDName(self.ref_dir_genome / "id_ens_canonical_ncrna.txt.gz",biotype='ncrna')
        elif biotype == 'pseudogene':
            return IDName(self.ref_dir_genome / "id_ens_canonical_pseudogene.txt.gz",biotype='pseudogene')
    
    def combine_annot(self,key):
        
        out = {}
        for biotype in self.biotype_list:
            annot = self.get_annot_gene(biotype,key)
            out.update(annot.annot_dict)
            biotypes_uniq = np.unique([x['gene_biotype'] for x in annot.annot_dict.values()])
            self.dict_biotypes[biotype] = biotypes_uniq.tolist()
        
        return out

    def combine_annot_tr(self):
        
        out = {}
        for biotype in self.biotype_list:
            annot = self.get_annot_tr(biotype)
            out.update(annot.annot_dict)
        
        return out
    
    def combine_id_tr(self):

        out = {}
        for biotype in self.biotype_list:
            id = self.get_id_tr(biotype)
            out.update(id.dict_name)
        return out

    def combine_seq_tr(self):

        out = {}
        for biotype in self.biotype_list:
            seq = self.get_seq_tr(biotype)
            out.update(seq.dict_seq)
        return out


class RefNC:

    def __init__(
        self,
        data_dir,
        ref_dir,
        sp,
        annot_key='tr_id'
        ):

        self.sp = sp
        self.biotype = 'ncrna'

        if data_dir != '':
            data_dir = Path(data_dir)
            self.data_dir = data_dir
            exp_metadata_file = data_dir / "exp_metadata_ncrna.csv"
            '''parse metadata of the experiment'''
            self.exp_metadata = ExpMetadata(exp_metadata_file)
            print("experimental metadata has been loaded...")

        if ref_dir != '':
            ref_dir = Path(ref_dir)
            self.ref_dir = ref_dir
            ref_ncrna_file = ref_dir / 'ref_ncrna_seq.fa.gz'
            
            self.ref_ncrna = RefcDNA(Path(ref_ncrna_file))
            print("reference cDNA has been loaded...")


class RefPseudo:

    def __init__(
        self,
        data_dir,
        ref_dir,
        sp,
        annot_key='tr_id'
        ):

        self.sp = sp
        self.biotype = 'pseudogene'

        if data_dir != '':
            data_dir = Path(data_dir)
            self.data_dir = data_dir
            exp_metadata_file = data_dir / "exp_metadata_pseudogene.csv"
            '''parse metadata of the experiment'''
            self.exp_metadata = ExpMetadata(exp_metadata_file)
            print("experimental metadata has been loaded...")

        if ref_dir != '':
            ref_dir = Path(ref_dir)
            self.ref_dir = ref_dir
            ref_pseudogene_file = ref_dir / 'ref_pseudogene_seq.fa.gz'
            
            self.ref_pseudogene = RefcDNA(Path(ref_pseudogene_file))
            print("reference cDNA has been loaded...")

