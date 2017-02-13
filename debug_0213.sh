echo "start training debug data set"
RESULTDIR=debug_emb
DATADIR=debug_data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/1200
time ./fasttext skipgram -input "${DATADIR}"/emr.txt -output "${RESULTDIR}"/debug100_ws16 \
-epoch 10 -thread 12 -ws 16 -beta_base 8 -delta 0.2 -neg 5 -t 2e-2 -dim 100
echo "trained 100 dim debug embedding"

