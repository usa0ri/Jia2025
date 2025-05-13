FROM python:3.9-slim

# install python packages
COPY requirements.txt requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y wget \
    fontconfig gcc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libegl-dev libxkbcommon-x11-0 libdbus-1-3 \
    qt6-base-dev libxcb-cursor0 libxcb-xinerama0

RUN wget http://ftp.br.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.8_all.deb && \
    apt --fix-broken install -y ./ttf-mscorefonts-installer_3.8_all.deb && \
    rm ttf-mscorefonts-installer_3.8_all.deb && \
    fc-cache -f -v

ENV QT_DEBUG_PLUGINS=1

###################
ARG USERNAME=saori
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME