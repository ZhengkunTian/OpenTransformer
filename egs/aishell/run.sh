###
 # @Author: Zhengkun Tian
 # @Email: zhengkun.tian@outlook.com
 # @Date: 2020-06-11 22:11:37
 # @LastEditTime: 2020-06-12 14:58:03
 # @FilePath: \OpenTransducer\egs\aishell\run.sh
### 
. ./path.sh || exit 1;

stage=-1

data=/data2/corpus/aishell
data_url=www.openslr.org/resources/33

set -e
set -u
set -o pipefail

if [ ${stage} -le -1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    sh local/download_and_untar.sh $data $data_url data_aishell || exit 1;
    sh local/download_and_untar.sh $data $data_url resource_aishell || exit 1;

    sh local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;

    for name in train test dev;do
        python local/split_and_norm.py data/$name/text.org data/$name/text || exit 1;
    done

    # # python local/prepare_data.py $datadir data || exit 1;
    # prepare vocabulary!
    python local/generate_vocab.py data/train/text data/vocab || exit 1;

fi