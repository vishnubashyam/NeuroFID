#! /bin/bash +x
#Script from Guray

tempImg=`pwd`/'../Templates/MNI152_T1_1mm_brain_LPS_filled.nii.gz'

listImg='../Lists/missing.csv'
outDir=`pwd`/'../BrainAligned'



i=1
for ll in `sed 1d $listImg`; do
#for ll in `sed 1d $listImg | head -20`; do

    echo "Subj $i ..."

    id=`echo $ll | cut -d, -f1`
    imgT1=`echo $ll | cut -d, -f8`
    imgB=`echo $ll | cut -d, -f10`


    outT1="${outDir}/${id}_T1_BrainAligned.nii.gz"

    if [ -f $imgT1 ] && [ -f $imgB ]; then
        mkdir -pv ${outDir}/sge_out
        if [ ! -f $outT1 ]; then
            qsub -l short -j y -b y -o ${outDir}/${std}/sge_out/\$JOB_NAME-\$JOB_ID.log `pwd`/my_flirt_wrapper.sh ${imgT1} ${imgB} ${tempImg} ${outT1}
        fi
    fi

    i=$(( i + 1 ))

#     read -p ee

done
