#! /bin/bash +x

# Apply brain mask to T1 image, linearly align to template, save as byte image

export FSLOUTPUTTYPE='NIFTI_GZ'

imgT1=$1
imgB=$2
imgTemp=$3
imgOut=$4

TMP=`mktemp -d -p ${SBIA_TMPDIR} my_flirt_XXXXXXXXXX`
#TMP='./ttt2'
#mkdir -pv $TMP

3dcalc -a $imgT1 -b $imgB -expr "a*b" -byte -prefix ${TMP}/bimg.nii.gz
flirt -in ${TMP}/bimg.nii.gz -ref ${imgTemp} -omat ${TMP}/bimg_fl.mat -out ${TMP}/bimg_fl.nii.gz -datatype short

cp ${TMP}/bimg_fl.nii.gz ${imgOut}
cp ${TMP}/bimg_fl.mat ${imgOut%.nii.gz}.mat

rm -rf ${TMP}
