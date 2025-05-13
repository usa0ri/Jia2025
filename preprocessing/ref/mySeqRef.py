from pathlib import Path
import pandas as pd
import re
import gzip
import numpy as np
import pickle
import re
import pprint

import myRiboSeq.mylib as my

GTF_HEADER  = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attr']
GTF_HEADER_dtypes = {
    'seqname':'str',
    'source':'str',
    'feature': 'str',
    'start':'int',
    'end': 'int',
    'score': 'str',
    'strand': 'str',
    'frame': 'str',
    'attr': 'str'
    }
GTF_HEADER_attr = ['gene_id','gene_version','gene_name','gene_source','gene_biotype',\
    'transcript_id','transcript_version','transcript_name','transcript_source','transcript_biotype',\
    'exon_id','exon_version',  'exon_number',\
    'ccds_id','protein_id','protein_version','tag','transcript_support_level']
cDNA_HEADER = ['transcript_id', 'transcript_id_v', 'chr', 'start', 'end', 'direction', 'gene_id',\
    'gene_biotype', 'tr_biotype', 'seq']
ORF_SDG_HEADER = ['gene_id','gene_name','chr','start','end','SDG_ID','seq']
DNA_HEADER = ['chr_name', 'start', 'end','seq']
transcriptGENCODE_HEADER = ['transcript_id', 'transcript_id_v', 'gene_id', 'gene_id_v',\
    'cds_start', 'cds_end', 'len_tr', 'direction', 'seq']
ncRNA_HEADER = ['transcript_id','transcript_id_v','chr','start','end','direction',\
    'gene_id','gene_id_v','gene_biotype','seq']

START_CODON = ['ATG', 'ATA', 'TTG','GTG','CTG','ATT','ATC']
STOP_CODON = ['TGA','TAA','TAG']

APPRIS_ORDERS = np.array(['PRINCIPAL:1','PRINCIPAL:2','PRINCIPAL:3','PRINCIPAL:4','PRINCIPAL:5',
    'ALTERNATIVE:1','ALTERNATIVE:2'
])

def rev(seq):
    return seq[::-1].translate(str.maketrans({'A':'T','T':'A','C':'G','G':'C'}))

class GTFdata():
    def __init__(
        self,
        gtf_file,
        attr_gtf,
        attr_grp=None,
        query_str='protein_coding'):

        self.gtf_file = gtf_file
        assert np.all(np.array(map(lambda x: x in GTF_HEADER_attr, attr_gtf)))
        print(f'\nloading {gtf_file} ...')
        
        result = {}
        df_ = pd.read_csv(
            gtf_file,
            compression='gzip',delimiter='\t',header=None,names=GTF_HEADER,dtype=GTF_HEADER_dtypes)
        df = df_[df_['attr'].str.contains(query_str)]
        for i,key in enumerate(attr_gtf):
            print(f'\r{i+1}/{len(attr_gtf)} attribute in .gtf file...', end='')
            result[key] = df['attr'].map(lambda x: re.findall(f'{key}\s\"(.*?)\".*?;',x)[0] if (key in x) else None)
            if (key in ['gene_id','transcript_id']) and ('ENS' in result[key].iloc[0]):
                result[key] = result[key].map(lambda x: x.split('.')[0])
        df_attr = pd.DataFrame(result,index=df.index)

        if attr_grp:
            self.df_gtf = pd.merge(df.iloc[:,:8],df_attr,left_index=True, right_index=True, how='inner')\
                .groupby(attr_grp)
        else:
            self.df_gtf = pd.merge(df.iloc[:,:8],df_attr,left_index=True, right_index=True, how='inner')
        print('\nGTF data has been loaded.')

class cDNAdata():
    def __init__(self,fa_file,attr_query):
        result = []
        fn_open = gzip.open if fa_file.suffix == '.gz' else open   
        with fn_open(fa_file) as fh:
            content = fh.read().decode('utf-8').split('>')[1:]
            tr_id_list = []
            n_cdna = len(content)
            for i,c in enumerate(content):
                print(f'\r{i+1}/{n_cdna} cdna in .fa file...', end='')
                if attr_query not in c:
                    continue
                seq = ''.join(c.split('\n')[1:]).strip()
                attr = c.split('\n')[0]
                tr_id_ = attr.split(' ')[0]
                if 'ENS' in tr_id_:
                    tr_id,tr_id_v = tr_id_.split('.')
                    gene_id,gene_id_v = re.match('.*gene:(ENS[A-Z]{1,4}[0-9]+)\.([0-9]*)',attr).group(1,2)
                elif 'WBGene' in attr:
                    tr_id = tr_id_
                    tr_id_v = '.'
                    gene_id = re.match('.*gene:(WBGene[0-9]+) ',attr).group(1)
                elif 'FBtr' in attr:
                    tr_id = tr_id_
                    tr_id_v = '.'
                    gene_id = re.match('.*gene:(FBgn[0-9]+) ',attr).group(1)

                chromosome, start, stop, strand = attr.split(' ')[2].split(':')[2:]
                
                df_now = [
                    tr_id,tr_id_v,
                    chromosome, int(start), int(stop), int(strand),
                    gene_id,# gene Ensembl ID
                    re.match('.*gene_biotype:(\w+)',attr).group(1),# gene_biotype
                    re.match('.*transcript_biotype:(\w+)',attr).group(1),# tr_biotype
                    ]
                df_now.append(seq)
                result.append(df_now)
                tr_id_list.append(df_now[0])

        self.df_seq = pd.DataFrame(result,columns=cDNA_HEADER,index=tr_id_list)
        print('\ncDNA data has been loaded.')

class transcriptGENCODEdata():
    def __init__(self,fa_file):
        result = []
        fn_open = gzip.open if fa_file.suffix == '.gz' else open   
        with fn_open(fa_file) as fh:
            content = fh.read().decode('utf-8').split('>')[1:]
            tr_id_list = []
            n_cdna = len(content)
            for i,c in enumerate(content):
                print(f'\r{i+1}/{n_cdna} cdna in .fa file...', end='')
                seq = ''.join(c.split('\n')[1:]).strip()
                attr =c.split('\n')[0].split('|')
                cds_start,cds_stop = re.match('.*\|CDS:([0-9]+)-([0-9]+)\|.*',c.split('\n')[0]).group(1,2)
                len_tr = re.match('.*\|([0-9]+)\|.*',c.split('\n')[0]).group(1)
                direction = -1 if '|-|' in c.split('\n')[0] else 1
                df_now = [
                    attr[0].split('.')[0],# transcript Ensembl ID
                    attr[0].split('.')[1],# transcript Ensembl ID version
                    attr[1].split('.')[0],# gene Ensembl ID
                    attr[1].split('.')[1],# gene Ensembl ID version
                    int(cds_start),int(cds_stop),
                    int(len_tr),# length
                    int(direction)
                    ]
                df_now.append(seq)
                result.append(df_now)
                tr_id_list.append(df_now[0])

        self.df_seq = pd.DataFrame(result,columns=transcriptGENCODE_HEADER,index=tr_id_list)
        print('\ntranscript GENCODE data has been loaded.')
    
class DNAdata():
    def __init__(self,fa_file):
        result = []
        fn_open = gzip.open if fa_file.suffix == '.gz' else open   
        with fn_open(fa_file) as fh:
            content = fh.read().decode('utf-8').split('>')[1:]
            n_cdna = len(content)
            chr_nums = []
            for i,c in enumerate(content):
                print(f'\r{i+1}/{n_cdna} chromosome in .fa file...', end='')
                seq = ''.join(c.split('\n')[1:]).strip()
                chro = c.split('\n')[0].strip().split()
                df_now = [
                    chro[1],# chromosome name
                    int(chro[2].split(':')[3]),# start
                    int(chro[2].split(':')[4]),# end
                    seq # sequence
                ]
                chr_nums.append(chr[0])
                result.append(df_now)

        self.df_seq = pd.DataFrame(result,columns=DNA_HEADER,index=chr_nums)
        print('\nDNA data has been loaded.')

class ncRNAdata():
    def __init__(self,fa_file):
        result = []
        fn_open = gzip.open if fa_file.suffix == '.gz' else open
        with fn_open(fa_file) as fh:
            content = fh.read().decode('utf-8').split('>')[1:]
            n_ncrna = len(content)
            for i,c in enumerate(content):
                print(f'\r{i+1}/{n_ncrna} ncRNA in .fa file...', end='')
                attr = c.split('\n')[0]
                tr_id,tr_id_v = attr.split(' ')[0].split('.')
                chromosome, start, stop, strand = attr.split(' ')[2].split(':')[2:]
                gene_id,gene_id_v = re.match('.*gene:(ENS[A-Z]{1,3}[0-9]+)\.([0-9]*)',attr).group(1,2)
                df_now = [
                    tr_id,tr_id_v,
                    chromosome, int(start), int(stop), int(strand),
                    gene_id,gene_id_v,
                    re.match('.*gene_biotype:(\w+)',attr).group(1),# gene_biotype
                    ''.join(c.split('\n')[1:]).strip()# seq
                ]
                result.append(df_now)
        self.df_seq = pd.DataFrame(result,columns=ncRNA_HEADER)
        print('\nncRNA data has been loaded.')

class MANEsummary():
    def __init__(self,summary_file):
        data = pd.read_csv(summary_file,skiprows=0,delimiter='\t')
        self.df_sum = data

class APPRISdata():
    def __init__(self,appris_path):
        data = pd.read_csv(appris_path,delimiter='\t',header=0,index_col=1)
        self.df_data = data

class myTranscriptomeRef:

    def __init__(
        self,
        id_file,
        id_file_refseq,
        cdna_encode_fa,
        annot_dir,
        appris_dir = '',
        cdna_gencode_fa = '',):

        df_id = pd.read_csv(
            id_file,sep="\t",
            names=["gene_id","ncbi_id","symbol"],
            dtype=str,
            index_col=1,skiprows=0,header=0)
        df_id2 = pd.read_csv(
            id_file_refseq,
            sep="\t",
            names=["gene_id","gene_id_ver","transcript_id","transcript_id_ver","refseq_mRNA"],
            dtype=str,
            index_col=2,skiprows=0,header=0)  

        self.df_id = pd.merge(df_id,df_id2,left_index=True, right_index=True, how='outer')
        self.data_dir = Path(annot_dir)
        self.cdna_gencode_fa = Path(cdna_gencode_fa)
        self.cdna_encode_fa = Path(cdna_encode_fa)
        self.appris_dir = Path(appris_dir)
        
    def _check_exon_cdna_all(self,i,tr,df_gtf,cdna_tr,dna,n_tr):
        print(f'\r{i+1}/{n_tr} transcript ...', end='')
        tr_exon_info = df_gtf.get_group(tr).reset_index()
        chr_now = tr_exon_info['seqname'][0]
        # make transcript sequence by combining all the exon sequences
        tr_seq = [
            dna.df_seq.loc[str(chr_now),'seq'][ tr_exon_info['start'][j]-1 : tr_exon_info['end'][j] ]
            for j in np.argsort(np.array(tr_exon_info['start']))
        ]
        if tr_exon_info['strand'][0] == '-':
            tr_seq_out = rev(''.join(tr_seq))
        elif tr_exon_info['strand'][0] == '+':
            tr_seq_out = ''.join(tr_seq)
        cdna_seq = cdna_tr['seq']

        flag = (tr_seq_out == cdna_seq) and (cdna_tr['start'] == np.min(tr_exon_info['start'])) and (cdna_tr['end'] == np.max(tr_exon_info['end']))
        return flag

    def check_exon_cdna(self,tr_list,save_dir):
        save_dir = Path(save_dir)
        cdna = cDNAdata( self.data_dir / "Homo_sapiens.GRCh38.cdna.all.fa.gz"  ,"protein_coding" )
        dna = DNAdata( self.data_dir / f'Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz' )
        gtf_exon = GTFdata( self.data_dir / "extract/exon.gz" , ['gene_id','transcript_id'] , attr_grp='transcript_id')
        
        if len(tr_list) == 0:
            tr_list = list(gtf_exon.df_gtf.size().index)

        flags = list(map(
            lambda x: self._check_exon_cdna_all(
                x[0],
                x[1],
                gtf_exon.df_gtf,
                cdna.df_seq.loc[x[1],:],
                dna,
                len(tr_list)
                ),enumerate(tr_list)))
        assert np.all(np.array(flags))
    
    def _parse_cdna(
        self,
        tr,
        gene_cds,
        gene_exon,
        cdna,
        gtf_start_gene,
        gtf_stop_gene):

        len_cds = gene_cds.query(f'transcript_id == "{tr}"')\
            .groupby(['start','end'])\
                .apply(lambda x: x.end - x.start + 1).sum()
        if type(len_cds) is not np.int64:
            len_cds = len_cds.iloc[0]
        
        len_cdna = gene_exon.query(f'transcript_id == "{tr}"')\
            .groupby(['start','end'])\
                .apply(lambda x: x.end - x.start + 1).sum()
        if type(len_cdna) is not np.int64:
            len_cdna = len_cdna.iloc[0]
        len_exon_ = gene_exon.query(f'transcript_id == "{tr}"')\
            .groupby(['start','end'])\
                .apply(lambda x: x.end - x.start + 1)
        if len(len_exon_) == 1:
            len_exon_ = len_exon_.iloc[0]

        gene_start_codon = gtf_start_gene.query(f'transcript_id == "{tr}"')['start'].iloc[0]
        gene_stop_codon = gtf_stop_gene.query(f'transcript_id == "{tr}"')['start'].iloc[0]

        if cdna['direction'] == 1:
            
            if len(len_exon_) == 1:
                len_exon = len_exon_
            else:
                len_exon = len_exon_.values
            
            # to calculate start_cds, get intron length before start_tr
            gene_start_tr = gene_cds.query(f'transcript_id == "{tr}"')['start'].min()
            idx_exon = np.where(
                (gene_start_tr <= gene_exon.query(f'transcript_id == "{tr}"')['end']) \
                    * (gene_start_tr >= gene_exon.query(f'transcript_id == "{tr}"')['start']))[0]
            introns = 0
            for ex in range(int(idx_exon)):
                introns += gene_exon.query(f'transcript_id == "{tr}"')['start'].iloc[ex+1] \
                    - gene_exon.query(f'transcript_id == "{tr}"')['end'].iloc[ex] -1
            
            assert len_cdna == np.sum(len_exon)
            pos = [
                gene_start_codon,# start_codon_gene
                gene_start_tr - int(cdna.loc['start']) - introns,# start_cds
                gene_stop_codon, # stop_codon_gene
                len_cds + gene_start_tr - int(cdna.loc['start']) - introns,# stop_cds
                len_cdna # len_cdna
            ]

        elif cdna['direction'] == -1:

            if len(len_exon_) == 1:
                len_exon = len_exon_[::-1]
            else:
                len_exon = len_exon_.values[::-1]
            
            # to calculate start_cds, get intron length after start_tr
            gene_start_tr = gene_cds.query(f'transcript_id == "{tr}"')['end'].max()
            idx_exon = np.where(
                (gene_start_tr <= gene_exon.query(f'transcript_id == "{tr}"')['end']) \
                    * (gene_start_tr >= gene_exon.query(f'transcript_id == "{tr}"')['start']))[0]
            introns = 0
            for ex in range(int(idx_exon)):
                introns += gene_exon.query(f'transcript_id == "{tr}"')['start'].iloc[ex] \
                    - gene_exon.query(f'transcript_id == "{tr}"')['end'].iloc[ex+1] -1
            
            assert len_cdna == np.sum(len_exon)
            pos = [
                gene_start_codon,# start_codon_gene
                int(cdna.loc['end']) - gene_start_tr - introns,# start_cds
                gene_stop_codon, # stop_codon_gene
                len_cds + int(cdna.loc['end']) - gene_start_tr - introns,# stop_cds
                len_cdna # len_cdna
            ]
            
        flag = np.array([
            (cdna['seq'][pos[1]:pos[1]+3] in START_CODON),
            (cdna['seq'][pos[3]:pos[3]+3] in STOP_CODON)])
        return flag,pos,len_exon

    def select_cdna(
        self,
        save_dir,
        ref_source):

        save_dir = save_dir / f'select_cdna_{ref_source}'
        if not save_dir.exists():
            save_dir.mkdir()
        
        output_failed = Path(save_dir / 'failed_genes.txt.gz')
        output_failed.unlink(missing_ok=True)

        if 'hsa' in self.appris_dir.stem:
            df_appris = pd.read_csv(
                self.appris_dir / 'appris_data.principal.txt.gz',
                sep='\t',index_col=1,header=None
            ).set_axis(['symbol','transcript_id','ccds_id','appris_pr','mane'],axis=1)
        else:
            df_appris = pd.read_csv(
                self.appris_dir / 'appris_data.principal.txt.gz',
                sep='\t',index_col=1,header=None
            ).set_axis(['symbol','transcript_id','strand','appris_pr'],axis=1)
        
        if ref_source == 'encode':
            cdna = cDNAdata(self.cdna_encode_fa,"protein_coding")
        elif ref_source == 'gencode':
            cdna = transcriptGENCODEdata(self.cdna_gencode_fa)
            cdna_ens = cDNAdata(self.cdna_encode_fa,"protein_coding")
        gtf_exon = GTFdata(
            self.data_dir / "extract/exon.gz" ,
            ['gene_id','transcript_id','ccds_id'] ,
            attr_grp='gene_id')
        gtf_cds = GTFdata(
            self.data_dir / "extract/CDS.gz" ,
            ['gene_id','transcript_id','ccds_id', 'gene_biotype', 'transcript_biotype' ],
            attr_grp='gene_id')
        gtf_start_codon = GTFdata(
            self.data_dir / "extract/start_codon.gz" ,
            ['gene_id','transcript_id','ccds_id'],
            attr_grp='gene_id')
        gtf_stop_codon = GTFdata(
            self.data_dir / "extract/stop_codon.gz" ,
            ['gene_id','transcript_id','ccds_id'],
            attr_grp='gene_id')
        
        name_gtfs = np.array(['exon','cds','start_codon','stop_codon'])
        gene_list_gtf = [
            list(gtf_exon.df_gtf.size().index),
            list(gtf_cds.df_gtf.size().index),
            list(gtf_start_codon.df_gtf.size().index),
            list(gtf_stop_codon.df_gtf.size().index)
        ]
        gene_list_fa = cdna.df_seq['gene_id'].unique()
        gene_list = gene_list_fa
        for genes in gene_list_gtf:
            gene_list = list( set(genes) & set(list(gene_list)) )
        gene_list_onlyfa =list( set(gene_list_fa) - set(list(gene_list)) )
        print(f'{len(gene_list_onlyfa)}/{len(gene_list_fa)} transcripts did not have annotations in GTF files')
        with gzip.open(output_failed,mode='at',encoding='utf-8') as f:
            for gene in gene_list_onlyfa:
                is_included = np.array([gene in gtf for gtf in gene_list_gtf])
                name_str = ';'.join(name_gtfs[~is_included])
                f.write(f'{gene}\tnot_gtf:{name_str}\n')

        output_exception = Path(save_dir / 'exception_transcripts.txt.gz')
        output_exception.unlink(missing_ok=True)
        # Follow GFF 3 format
        attrs = [
            'ref',# transcript id
            'source',# gene id
            'type',
            'start',# within transcripts
            'end',# within transcripts,
            'score',
            'strand',
            'phase',
            'attrs'# 'len_cdna','len_exon','start_codon_gene','stop_codon_gene',
            ]
        pos_list = {};pos_list2 = {}
        for attr in attrs:
            pos_list[attr] = [];pos_list2[attr] = []
        tr_info_list = {}
        counts_tmp = {'1':0,'>1':0}
        for j,gene in enumerate(gene_list):
            print(f'\r{j+1}/{len(gene_list)} gene...', end='')

            gene_cds = gtf_cds.df_gtf.get_group(gene)
            gene_exon = gtf_exon.df_gtf.get_group(gene)
            gtf_start_gene = gtf_start_codon.df_gtf.get_group(gene)
            gtf_stop_gene = gtf_stop_codon.df_gtf.get_group(gene)

            seqnames = gene_exon['seqname'].unique()
            if len(seqnames) > 1:
                seqnames_str = ' and '.join(seqnames)
                # FIXME
                print(f'{tr} locates both in {seqnames_str}. choose {seqnames[0]}...')
                # some transcripts (ENST00000390665 etc) locate on chrX and chrY with same splicing sites
                gene_cds = gene_cds.query(f'seqname == "{seqnames[0]}"')
                gene_exon = gene_exon.query(f'seqname == "{seqnames[0]}"')
                gtf_start_gene = gtf_start_gene.query(f'seqname == "{seqnames[0]}"')
                gtf_stop_gene = gtf_stop_gene.query(f'seqname == "{seqnames[0]}"')
            
            tr_list = np.array(list(
                set(gene_cds['transcript_id'].unique()) &\
                    set(gene_exon['transcript_id'].unique()) &\
                        set(gtf_start_gene['transcript_id'].unique()) &\
                            set(gtf_stop_gene['transcript_id'].unique())
            ))
            
            # select transcripts with APPRIS score
            # -> transcript with the longest CDS
            if gene == "ENSG00000128272":
                # For ATF4, select ENST00000674920, not the longest ENST00000337304
                tr = "ENST00000674920"
                if ref_source == 'gencode':
                    cdna_tr = cdna_ens.df_seq.loc[tr,:]
                elif ref_source == 'encode':
                    cdna_tr = cdna.df_seq.loc[tr,:]
                flag,pos,len_exon = self._parse_cdna(tr,gene_cds,gene_exon,cdna_tr,gtf_start_gene,gtf_stop_gene)
            elif len(tr_list) == 0:
                print(f'no protein-coding transcripts available for {gene}')
                with gzip.open(output_failed,mode='at',encoding='utf-8') as f:
                    f.write(f'{gene}\tno_transcripts\n')
                continue
            elif len(tr_list) == 1:
                tr = tr_list[0]
                counts_tmp['1'] += 1
                # cdna_tr = cdna.df_seq.loc[tr,:]
                if ref_source == 'gencode':
                    cdna_tr = cdna_ens.df_seq.loc[tr,:]
                elif ref_source == 'encode':
                    cdna_tr = cdna.df_seq.loc[tr,:]
                flag,pos,len_exon = self._parse_cdna(tr,gene_cds,gene_exon,cdna_tr,gtf_start_gene,gtf_stop_gene)
            else:# if multiple transcripts exist

                if gene in df_appris.index:
                    df_appris_now = df_appris.loc[gene,:]
                    if type(df_appris_now) is pd.Series:
                        df_appris_now = df_appris_now.to_frame().T   
                    idx_tr_appris = np.array([tr in df_appris_now['transcript_id'].values for tr in tr_list])
                    if np.any(idx_tr_appris):
                        df_appris_now.set_index('transcript_id',inplace=True)
                        tr_list = tr_list[idx_tr_appris]
                        is_appris = np.array([
                            np.where(APPRIS_ORDERS == df_appris_now.loc[tr,'appris_pr'])[0][0]
                            for tr in tr_list
                        ])
                        min_appris = is_appris.min()
                        tr_list = tr_list[ is_appris == min_appris ]
                else:
                    print(f"no APPRIS data for {gene}...")
                
                if len(tr_list) == 1:
                    tr = tr_list[0]
                    counts_tmp['1'] += 1
                    # cdna_tr = cdna.df_seq.loc[tr,:]
                    if ref_source == 'gencode':
                        cdna_tr = cdna_ens.df_seq.loc[tr,:]
                    elif ref_source == 'encode':
                        cdna_tr = cdna.df_seq.loc[tr,:]
                    flag,pos,len_exon = self._parse_cdna(tr,gene_cds,gene_exon,cdna_tr,gtf_start_gene,gtf_stop_gene)

                else:
                    
                    # remove duplicated coordinates
                    gene_cds = gene_cds.iloc[~gene_cds[['start','end','transcript_id']].duplicated().values,:]
                    len_cds_list_ = np.array([
                        gene_cds.query(f'transcript_id == "{tr}"')\
                            .apply(lambda x: x.end - x.start + 1,axis=1)\
                                .sum(axis=0)
                        for tr in tr_list
                    ],dtype=np.int32).reshape(len(tr_list))

                    # remove duplicated coordinates
                    gene_exon = gene_exon.iloc[~gene_exon[['start','end','transcript_id']].duplicated().values,:]
                    len_cdna_list_ = np.array([
                        gene_exon.query(f'transcript_id == "{tr}"')\
                            .apply(lambda x: x.end - x.start + 1,axis=1)\
                                .sum(axis=0)
                        for tr in tr_list
                    ],dtype=np.int32).reshape(len(tr_list))
                
                    # remove len_cds%3 != 0
                    idx_tr_ = len_cds_list_%3 == 0
                    len_cds_list = len_cds_list_[idx_tr_]
                    len_cdna_list = len_cdna_list_[idx_tr_]

                    idx_cds = np.array([i for i, x in enumerate(len_cds_list) if x == np.max(len_cds_list)])
                    if len(idx_cds) == 0:
                        print(f'{gene} no CDS showed multiples of 3...skipped')
                        continue
                    elif len(idx_cds) == 1:
                        tr = tr_list[idx_cds[0]]
                    else:# if >=1 transcript has the same CDS length, select the transcript with the longest cDNA
                        idx_tr = np.array([i for i, x in enumerate(len_cdna_list) if x == np.max(len_cdna_list)])
                        if len(idx_tr) == 1:
                            tr = tr_list[idx_cds[idx_tr[0]]]
                        else:
                            tr = tr_list[idx_cds[idx_tr[0]]]
                            tr_now = ';'.join(tr_list[idx_cds[idx_tr]])
                            print(f'{gene} has multiple transcripts with the same CDS and cDNA length:{tr_now}')
                            with gzip.open(output_exception,mode='at',encoding='utf-8') as f:
                                f.write(f'{tr}\t{gene}\tmultiple_transcripts={tr_now}\n')
                
                    # cdna_tr = cdna.df_seq.loc[tr,:]
                    if ref_source == 'gencode':
                        cdna_tr = cdna_ens.df_seq.loc[tr,:]
                    elif ref_source == 'encode':
                        cdna_tr = cdna.df_seq.loc[tr,:]
                    flag,pos,len_exon = self._parse_cdna(tr,gene_cds,gene_exon,cdna_tr,gtf_start_gene,gtf_stop_gene)
                    
                    counts_tmp['>1'] += 1

                    # transcripts not selected
                    for tr_ in tr_list[tr_list!=tr]:
                        if ref_source == 'gencode':
                            cdna_tr_ = cdna_ens.df_seq.loc[tr_,:]
                        elif ref_source == 'encode':
                            cdna_tr_ = cdna.df_seq.loc[tr_,:]
                        flag_,pos_,len_exon_ = self._parse_cdna(tr_,gene_cds,gene_exon,cdna_tr_,gtf_start_gene,gtf_stop_gene)
                        len_exon_str_ = ','.join([str(x) for x in len_exon_])
                        pos_list2['ref'].append(tr_)
                        pos_list2['source'].append(gene)
                        pos_list2['type'].append('.')
                        pos_list2['start'].append(pos_[1])
                        pos_list2['end'].append(pos_[3]+3)
                        pos_list2['score'].append('.')
                        pos_list2['strand'].append('+')
                        pos_list2['phase'].append('.')
                        pos_list2['attrs'].append(';'.join([
                            f'len_cdna={pos_[4]}',
                            f'len_exon={len_exon_str_}',
                            f'start_gene={pos_[0]}',
                            f'end_gene={pos_[2]}'
                        ]))

            if np.any(~flag):
                if not flag[0]:
                    start_codon_now = cdna_tr['seq'][pos[1]:pos[1]+3]
                    print(f'{tr} of {gene} has start codon of {start_codon_now}')
                    with gzip.open(output_exception,'at') as f:
                        f.write(f'{tr}\t{gene}\tstart_codon={start_codon_now}\n')
                if not flag[1]:
                    stop_codon_now = cdna_tr['seq'][pos[3]:pos[3]+3]
                    print(f'{tr} of {gene} has stop codon of {stop_codon_now}')
                    with gzip.open(output_exception,'at') as f:
                        f.write(f'{tr}\t{gene}\tstop_codon={stop_codon_now}\n')

            if type(pos[4]) is not np.int64:
                pos[4] = ','.join([str(x) for x in pos[4]])
            len_exon_str = ','.join([str(x) for x in len_exon])
            pos_list['ref'].append(tr)
            pos_list['source'].append(gene)
            pos_list['type'].append('.')
            pos_list['start'].append(pos[1]+1)
            pos_list['end'].append(pos[3]+1+3)
            pos_list['score'].append('.')
            pos_list['strand'].append('+')
            pos_list['phase'].append('.')
            pos_list['attrs'].append(';'.join([
                f'len_cdna={pos[4]}',
                f'len_exon={len_exon_str}',
                f'start_gene={pos[0]+1}',
                f'end_gene={pos[2]+1}'
            ]))

            # transcript information
            tmp =  self.df_id.loc[tr,'symbol']
            tmp2 = self.df_id.loc[tr,'refseq_mRNA']
            if type(tmp) is pd.Series:
                tmp_symbol = tmp.unique()[0]
                tmp_refseq = tmp2.unique()[0]
            elif type(tmp) is str:
                tmp_symbol = tmp
                tmp_refseq = tmp2
            elif type(tmp) is float:
                tmp_symbol = tmp
                tmp_refseq = tmp2
            tr_info_list[tr] = [tmp_symbol,tmp_refseq]
        
        outfile_name = save_dir / 'annot_selected_transcripts.gff.gz'
        pd.DataFrame(pos_list).to_csv(outfile_name,sep='\t',index=False,header=False)
        outfile_name = save_dir / 'annot_nonselected_transcripts.gff.gz'
        pd.DataFrame(pos_list2).to_csv(outfile_name,sep='\t',index=False)
        with open(save_dir / 'tr_list.pkl',"wb") as f:
            pickle.dump(pos_list['ref'], f)
        with open(save_dir / 'tr_list_notselected.pkl',"wb") as f:
            pickle.dump(pos_list2['ref'], f)
        pd.DataFrame().from_dict(tr_info_list,orient='index')\
            .set_axis(['HGNC_symbol','refseq_mRNA'],axis=1)\
            .to_csv(save_dir / 'tr_info.txt.gz',header=False,sep='\t')
    
    def output_cdna_fasta(self,save_dir,load_pkl,fname):

        with open(load_pkl, 'rb') as f:
            tr_list = pickle.load(f)
        
        cdna = cDNAdata( self.cdna_encode_fa ,"protein_coding" )
        
        with gzip.open((save_dir / fname).with_suffix('.fa.gz'), 'wt') as f:
            for tr in tr_list:
                seq = cdna.df_seq.loc[tr,'seq']
                f.write(f'>{tr}\n{seq}\n')
    
    def get_ncrna(self,save_dir,ncrna_fa):
        save_dir = save_dir / 'get_ncrna'
        if not save_dir.exists():
            save_dir.mkdir()

        if 'ensembl' in ncrna_fa:
            ncrna = ncRNAdata( Path(ncrna_fa) )
            ncrna.df_seq['gene_biotype'].value_counts()\
                .to_csv(save_dir / 'gene_biotype.csv.gz',compression='gzip')
        elif 'gencode' in ncrna_fa:
            ncrna = transcriptGENCODEdata( Path(ncrna_fa), '' )

    def check_overlap_genes(self,save_dir,tr_list):
        save_dir = save_dir / 'check_overlap_genes'
        if not save_dir.exists():
            save_dir.mkdir()

        gtf_exon = GTFdata(
            self.data_dir / "extract/exon.gz" ,
            ['gene_id','transcript_id'] ,
            attr_grp='seqname')
        
        seqnames = list(gtf_exon.df_gtf.size().index)
        for seqname in seqnames:
            tmp = gtf_exon.df_gtf.get_group(seqname)
            min_pos = tmp[['start','end']].min().min()
            max_pos = tmp[['start','end']].max().max()
            # detect overlap in regions divided by 100 bins

def _load_fa(
    fa_file
):
    dict_fa = {}
    with gzip.open(fa_file,mode='rt') as f:
        contents = ''.join(f.readlines()).split('>')
        for c in contents:
            if len(c.strip())<5:
                continue
            cs = c.split('\n')
            tr_id = cs[0]
            seq = ''.join(cs[1:])
            dict_fa[tr_id] = seq
    return dict_fa
