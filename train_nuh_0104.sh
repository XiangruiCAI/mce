echo "start training nuh data set"
RESULTDIR=nuh_emb_20180104_night2
DATADIR=~/datasets/nuh_data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

make

#./med2vec attn1 -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh_attn1_lr0.025_ws50_aws20_neg50 \
#-epoch 30 -thread 20 -neg 5 -t 1e-3 -dim 100 -lr 0.025 -ws 50 -attnws 20

# average frequency is about 1/400
for ws in 10 20 30 40 50
do
    for aws in 10 20 30
    do
        echo train on ws=$ws and aws=$aws with attn1
        ./med2vec attn1 -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh_attn1_ws$ws\_aws$aws\_epoch50 \
        -epoch 50 -thread 20 -neg 5 -t 1e-4 -dim 100 -lr 0.025 -ws $ws -attnws $aws

        echo train on ws=$ws and aws=$aws with attn2
        ./med2vec attn2 -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh_attn2_ws$ws\_aws$aws\_epoch50 \
        -epoch 50 -thread 20 -neg 5 -t 1e-4 -dim 100 -lr 0.025 -ws $ws -attnws $aws
    done
done
