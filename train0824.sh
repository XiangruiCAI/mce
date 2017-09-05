echo "start training nuh data set"
RESULTDIR=nuh_emb
DATADIR=~/datasets/nuh_data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/400
./med2vec attn1 -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh_attn1 \
-epoch 10 -thread 8 -neg 5 -t 1e-3 -dim 100 -nrand 8
echo "finish attention model 1 (context view)"

./med2vec attn2 -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh_attn1 \
-epoch 10 -thread 8 -neg 5 -t 1e-3 -dim 100 -nrand 8
echo "finish attention model 2 (context view)"
