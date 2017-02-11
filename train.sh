echo "start training nuh data set"
RESULTDIR=nuh_emb
DATADIR=nuh_data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/400
time ./fasttext skipgram -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh100 \
-epoch 10 -thread 12 -ws 30 -beta_base 20 -delta 0.2 -neg 5 -t 1e-2 -dim 100
echo "trained 100 dim nuh embedding"

time ./fasttext skipgram -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh200 \
-epoch 10 -thread 12 -ws 30 -beta_base 20 -delta 0.2 -neg 5 -t 1e-2 -dim 200
echo "trained 200 dim nuh embedding"

time ./fasttext skipgram -input "${DATADIR}"/nuh.txt -output "${RESULTDIR}"/nuh300 \
-epoch 10 -thread 12 -ws 30 -beta_base 20 -delta 0.2 -neg 5 -t 1e-2 -dim 300
echo "trained 300 dim nuh embedding"


echo "start training mimic data set"
RESULTDIR=mimic_emb
DATADIR=mimic

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/20000
time ./fasttext skipgram -input "${DATADIR}"/mimic.txt -output "${RESULTDIR}"/mimic100 \
-epoch 20 -thread 12 -ws 24 -beta_base 12 -delta 0.2 -neg 5 -t 1e-4 -dim 100 \
-timeUnit hour
echo "trained 100 dim mimic embedding"

time ./fasttext skipgram -input "${DATADIR}"/mimic.txt -output "${RESULTDIR}"/mimic200 \
-epoch 20 -thread 12 -ws 24 -beta_base 12 -delta 0.2 -neg 5 -t 1e-4 -dim 200 \
-timeUnit hour
echo "trained 200 dim mimic embedding"

time ./fasttext skipgram -input "${DATADIR}"/mimic.txt -output "${RESULTDIR}"/mimic300 \
-epoch 20 -thread 12 -ws 24 -beta_base 12 -delta 0.2 -neg 5 -t 1e-4 -dim 300 \
-timeUnit hour
echo "trained 300 dim mimic embedding"


echo "start training synpuf data set"
RESULTDIR=sample1_emb
DATADIR=sample_1_results

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# average frequency is about 1/1200
time ./fasttext skipgram -input "${DATADIR}"/emr.txt -output "${RESULTDIR}"/emr100 \
-epoch 10 -thread 12 -ws 50 -beta_base 25 -delta 0.2 -neg 5 -t 2e-2 -dim 100 \
echo "trained 100 dim sample1 embedding"

time ./fasttext skipgram -input "${DATADIR}"/emr.txt -output "${RESULTDIR}"/emr200 \
-epoch 10 -thread 12 -ws 50 -beta_base 25 -delta 0.2 -neg 5 -t 2e-2 -dim 200 \
echo "trained 200 dim sample1 embedding"

time ./fasttext skipgram -input "${DATADIR}"/emr.txt -output "${RESULTDIR}"/emr300 \
-epoch 10 -thread 12 -ws 50 -beta_base 25 -delta 0.2 -neg 5 -t 2e-2 -dim 300 \
echo "trained 300 dim sample1 embedding"

