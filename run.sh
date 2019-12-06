#pip install --editable .

#export CALIB=1 # for int8 calibration
export INT8=1 # to run int8 model

# profiling
# export SAMPLE_PROFILE=1 # profile each iteration
# export GLOBAL_PROFILE=1 # profile the total iteration
# OUTPUT_DIR=.

fairseq-generate \
	/home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.newstest2014 \
   	--path /home/huiwu1/workspace/transform/data-bin/wmt16.en-de.joined-dict.transformer/model.pt \
   	--beam 5 \
	--batch-size 128 \
	--remove-bpe 
