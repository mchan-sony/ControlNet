export PYTHONPATH=$PYTHONPATH:$(pwd)
export EXPERIMENT_NAME="day_and_night"
export MODE="test"

if [[ $MODE == "summary" ]]; then
    # Generate an image grid of results for high-level analysis
    export SUMMARY_FLAGS="--num_samples 32 \
        --batch_size 16"
    python3 $(pwd)/canny2image.py \
        $SUMMARY_FLAGS
elif [[ $MODE == "test" ]]; then
    # Generate individual images for detailed analysis
    export TEST_FLAGS="--num_samples 2048 \
        --batch_size 16"
    python3 $(pwd)/canny2image.py \
        $TEST_FLAGS
    # Print PSNR/SSIM metrics
    echo "Day-to-night faithfulness"
    python3 $(pwd)/../util/compute_faithfulness.py \
        $(pwd)/experiments/day_and_night/output/fakeB \
        $(pwd)/experiments/day_and_night/output/realA

    # Print FID metrics
    echo "Night realism"
    python3 -m pytorch_fid \
        $(pwd)/experiments/day_and_night/output/fakeB \
        $(pwd)/experiments/day_and_night/output/realB \
        --device cuda:0
fi

