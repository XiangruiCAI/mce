echo "start training nuh data set"
RESULTDIR=nuh_emb
DATADIR=~/datasets/nuh_data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/400
./med2vec cbow -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh100 \
-epoch 1 -thread 8 -neg 5 -t 1e-3 -dim 100 -nrand 8
echo "trained 100 dim nuh embedding"

