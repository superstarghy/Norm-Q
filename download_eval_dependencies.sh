#!/usr/bin/env sh

LIB=eval_metrics/lib
mkdir -p $LIB

echo "Downloading..."

# spice
SPICE=SPICE-1.0.zip
wget https://panderson.me/images/$SPICE
unzip SPICE-1.0.zip -d $LIB/
rm -f $SPICE

#
bash $LIB/SPICE-1.0/get_stanford_models.sh
