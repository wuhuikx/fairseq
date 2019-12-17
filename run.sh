#pip install --editable .

# download dataset and the trained model
# https://github.com/pytorch/fairseq/blob/15bd9bebbb6d34c70cacdc69ac21df2bbd6c1afd/examples/translation/README.md
# transformer.wmt16.en-de

# profiling
# export SAMPLE_PROFILE=1 # profile each iteration
# export GLOBAL_PROFILE=1 # profile the total iteration
# OUTPUT_DIR=.

# fp32 inference
#export engine='mkldnn' # can use fbgemm or mkldnn engine for fp32 inference
#fairseq-generate \
#	/home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.newstest2014 \
#   	--path /home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.transformer/model.pt \
#   	--beam 5 \
#	--batch-size 128 \
#	--remove-bpe \

# int8 calibration
export engine='fbgemm' # can only use fbgemm engine for calibration
fairseq-generate \
	/home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.newstest2014 \
   	--path /home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.transformer/model.pt \
   	--beam 5 \
	--batch-size 1 \
	--remove-bpe \
        --calib 

# int8 inference
export engine='mkldnn' # can use fbgemm or mkldnn engine for int8 inference
fairseq-generate \
	/home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.newstest2014 \
   	--path /home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.transformer/model.pt \
   	--beam 5 \
	--batch-size 128 \
	--remove-bpe \
        --int8 

