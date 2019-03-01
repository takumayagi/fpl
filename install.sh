#! /bin/sh
#
# install.sh
# Copyright (C) 2019 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

CDIR=$PWD
mkdir 3rdparty
cd 3rdparty
git clone git@github.com:takumayagi/mllogger.git

cd mllogger
python setup.py install

cd $CDIR
