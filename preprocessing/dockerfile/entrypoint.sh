#!/bin/bash

# user ID and group ID of docker
export USER=saori
export HOME=/home/$USER

# user ID and group ID of host
uid=$(stat -c "%u" .)
gid=$(stat -c "%g" .)

if [ "$uid" -ne 0 ]; then
    if [ "$(id -g $USER)" -ne $gid ]; then
        getent group $gid >/dev/null 2>%1 || groupmod -g $gid $USER
        chgrp -R $gid $HOME
    fi
    if [ "$(id -u $USER)" -ne $uid ]; then
        useromd -u $uid $USER
    fi
fi

exec setpriv --reuid=$USER --regid=$USER --init-groups "$@"