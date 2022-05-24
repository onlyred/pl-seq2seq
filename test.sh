#!/bin/sh

SEED=124
LOGO="./logs"
MODE="test"
EPOCHS=10
INPUT_DIM=8014	
OUTPUT_DIM=6191
EMB_DIM=256
HID_DIM=512
NLAYERS=2
MODEL_PATH="./logs/seq2seq/version_138/"
SAVED_MODEL=`ls ${MODEL_PATH}/*.ckpt`

python trainer.py --gpus '0'        \
                  --seed ${SEED}  \
		  --logdir ${LOGO}\
		  --epochs ${EPOCHS} \
		  --mode ${MODE} \
		  --input_dim ${INPUT_DIM} \
		  --output_dim ${OUTPUT_DIM} \
		  --embed_dim ${EMB_DIM} \
		  --hidden_dim ${HID_DIM} \
		  --nlayers ${NLAYERS} \
		  --saved_model ${SAVED_MODEL}
