
# ========== 配置区 ==========
# 请根据实际情况修改以下变量
MODEL_PATH=${MODEL_PATH:-/path/to/your/model/checkpoint}
ANSWER_FILE=${ANSWER_FILE:-your_answer_file.jsonl}
NUM_GPU=${NUM_GPU:-0}
NUM_GPU_WE_MATH=${NUM_GPU_WE_MATH:-1}

# 数据集路径变量
MM_MATH_QUESTION_FILE=${MM_MATH_QUESTION_FILE:-/path/to/mm_math/test_MM_Math.json}
MM_MATH_IMAGE_FOLDER=${MM_MATH_IMAGE_FOLDER:-/path/to/mm_math/images}
MM_MATH_ANSWER_FOLDER=${MM_MATH_ANSWER_FOLDER:-/path/to/mm_math/answers}

MATHVISTA_QUESTION_FILE=${MATHVISTA_QUESTION_FILE:-/path/to/mathvista/question.json}
MATHVISTA_IMAGE_FOLDER=${MATHVISTA_IMAGE_FOLDER:-/path/to/mathvista/images}
MATHVISTA_ANSWER_FOLDER=${MATHVISTA_ANSWER_FOLDER:-/path/to/mathvista/answers}

GEOQA_QUESTION_FILE=${GEOQA_QUESTION_FILE:-/path/to/geoqa/test.json}
GEOQA_IMAGE_FOLDER=${GEOQA_IMAGE_FOLDER:-/path/to/geoqa/images}
GEOQA_ANSWER_FOLDER=${GEOQA_ANSWER_FOLDER:-/path/to/geoqa/answers}

WEMATH_QUESTION_FILE=${WEMATH_QUESTION_FILE:-/path/to/wemath/test.json}
WEMATH_IMAGE_FOLDER=${WEMATH_IMAGE_FOLDER:-/path/to/wemath/images}
WEMATH_ANSWER_FOLDER=${WEMATH_ANSWER_FOLDER:-/path/to/wemath/answers}

CACHE_DIR=${CACHE_DIR:-/path/to/cache_dir}
CONV_MODE=${CONV_MODE:-qwen_1_5}
NUM_CHUNKS=${NUM_CHUNKS:-2}
TEMPERATURE=${TEMPERATURE:-0}

# ========== 评测脚本 ==========
{
# mm_math
CUDA_VISIBLE_DEVICES=${NUM_GPU} python -m eval.multi_mm_math \
    --model-path "${MODEL_PATH}" \
    --question-file "${MM_MATH_QUESTION_FILE}" \
    --image-folder "${MM_MATH_IMAGE_FOLDER}" \
    --answers-file "${MM_MATH_ANSWER_FOLDER}/${ANSWER_FILE}" \
    --temperature ${TEMPERATURE} \
    --num-chunks ${NUM_CHUNKS} \
    --conv-mode ${CONV_MODE} \
    --cache-dir "${CACHE_DIR}"

# mathvista
CUDA_VISIBLE_DEVICES=${NUM_GPU} python -m eval.multi_mathvista \
    --model-path "${MODEL_PATH}" \
    --question-file "${MATHVISTA_QUESTION_FILE}" \
    --image-folder "${MATHVISTA_IMAGE_FOLDER}" \
    --answers-file "${MATHVISTA_ANSWER_FOLDER}/${ANSWER_FILE}" \
    --temperature ${TEMPERATURE} \
    --num-chunks ${NUM_CHUNKS} \
    --conv-mode ${CONV_MODE} \
    --cache-dir "${CACHE_DIR}"

# geoqa
CUDA_VISIBLE_DEVICES=${NUM_GPU} python -m eval.multi_geoqa \
    --model-path "${MODEL_PATH}" \
    --question-file "${GEOQA_QUESTION_FILE}" \
    --image-folder "${GEOQA_IMAGE_FOLDER}" \
    --answers-file "${GEOQA_ANSWER_FOLDER}/${ANSWER_FILE}" \
    --temperature ${TEMPERATURE} \
    --num-chunks ${NUM_CHUNKS} \
    --conv-mode ${CONV_MODE} \
    --cache-dir "${CACHE_DIR}"

}&
{
# we_math
CUDA_VISIBLE_DEVICES=${NUM_GPU_WE_MATH} python -m eval.multi_wemath \
    --model-path "${MODEL_PATH}" \
    --question-file "${WEMATH_QUESTION_FILE}" \
    --image-folder "${WEMATH_IMAGE_FOLDER}" \
    --answers-file "${WEMATH_ANSWER_FOLDER}/${ANSWER_FILE}" \
    --temperature ${TEMPERATURE} \
    --num-chunks ${NUM_CHUNKS} \
    --conv-mode ${CONV_MODE} \
    --cache-dir "${CACHE_DIR}"

}&