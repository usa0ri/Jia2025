FROM ubuntu:20.04

####################################
RUN useradd -m saori
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y build-essential python3-pip python3-dev wget less\
                    autoconf automake make gcc perl \
                    libncurses5-dev zlib1g-dev libbz2-dev liblzma-dev libcurl4-gnutls-dev libssl-dev \
  && apt-get clean && apt-get purge \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && cd

####################################
## libraries ##

# cutadapt, fastqc, HISAT2, bowtie
RUN apt-get update \
  && apt-get install -y cutadapt fastqc hisat2 samtools bowtie

# STAR
RUN wget https://github.com/alexdobin/STAR/archive/2.7.10a.tar.gz \
  && tar -xzf 2.7.10a.tar.gz \
  && cd STAR-2.7.10a/source \
  && make STAR \
  && cp STAR /usr/local/bin  \
  && cd

# samtools
RUN wget https://github.com/samtools/samtools/releases/download/1.15.1/samtools-1.15.1.tar.bz2 \
  && tar xvjf samtools-1.15.1.tar.bz2 \
  && cd samtools-1.15.1 \
  && make \
  && make install

####################################
COPY entrypoint.sh /
RUN chmod +x entrypoint.sh
WORKDIR /home
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
