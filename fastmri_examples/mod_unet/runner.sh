#!/bin/bash

# Flags
use_hf_flag=false
train_mod_flag=false
test_mod_flag=false
pretrain_inference_flag=false
eval_mod_model=false
eval_pretrained_model=false
repo_id="btoto3/fastmri-dl"
max_len=""


# Default parameters
challenge="singlecoil"
data_path="/path/to/knee"
output_path_pretrain="output_pretrain/"
output_path_mod="output_mod/"
mask_type="random"
pretrain_challenge="unet_knee_sc"
max_epochs=50

# Function to display help message
function display_help {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --data_path PATH          Path to the dataset (default: /path/to/knee)"
    echo "  --mask_type TYPE          Mask type for training/testing (default: random)"
    echo "  --challenge TYPE          Challenge type: singlecoil or multicoil (default: singlecoil)"
    echo "  --output_path_pretrain PATH        Path to save pretrain outputs (default: output_pretrain/)"
    echo "  --output_path_mod PATH        Path to save mod outputs (default: output_mod/)"
    echo "  -train-mod                Train the modified U-Net model"
    echo "  -test-mod                 Test the modified U-Net model"
    echo "  -eval-mod                 Evaluate the modified U-Net model"
    echo "  -pretrain-inference       Run inference using the pretrained U-Net model"
    echo "  -eval-pretrained          Evaluate the pretrained U-Net model"
    echo "  -full-mod                 Run the full modified model pipeline"
    echo "  -full-pretrained          Run the full pretrained model pipeline"
    echo "  -use_hf                  Use Hugging Face for data and not local"
    echo " --max_len INT                 Maximum length of the input sequence, only applicable to hf for quick testing"
    echo " --max_epochs INT              Maximum number of epochs to train the model (default: 50)"
    echo "  --help                    Display this help message and exit"
    echo
    echo "Examples:"
    echo "  $0 -train-mod -eval-mod --data_path /path/to/data --challenge singlecoil"
    echo "  $0 -pretrain-inference --output_path /path/to/output"
    echo
    exit 0
}


# Parse arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --data_path)
        echo "Data Path set to: $2"
        data_path=$2; shift;;
    --mask_type)
        echo "Mask Type set to: $2"
        mask_type=$2; shift;;
    --challenge)
        echo "Challenge set to: $2"
        challenge=$2; shift;;
    --output_path_pretrain)
        echo "Output Path set to: $2"
        output_path_pretrain=$2; shift;;
    --output_path_mod)
        echo "Output Path set to: $2"
        output_path_mod=$2; shift;;
    --max_len)
        echo "Max Len set to: $2"
        max_len=$2; shift;;
    --max_epochs)
        echo "Max Epochs set to: $2"
        max_epochs=$2; shift;;
    -train-mod)
        echo "Train modified model flag set"
        train_mod_flag=true;;
    -test-mod)
        echo "Test modified model flag set"
        test_mod_flag=true;;
    -eval-mod)
        echo "Evaluate modified model flag set"
        eval_mod_model=true;;
    -full-mod)
        echo "Full modified model pipeline flag set"
        train_mod_flag=true
        test_mod_flag=true
        eval_mod_model=true;;
    -pretrain-inference)
        echo "Pretrain inference flag set"
        pretrain_inference_flag=true;;
    -eval-pretrained)
        echo "Evaluate pretrained model flag set"
        eval_pretrained_model=true;;
    -full-pretrained)
        echo "Full pretrained model pipeline flag set"
        pretrain_inference_flag=true
        eval_pretrained_model=true;;
    -use_hf)
        echo "Using Hugging Face Transformers library"
        use_hf_flag=true;;
    *)
        echo "Unknown argument: $1"
        exit 1;;
  esac
  shift
done

# Train modified model
if [ "$train_mod_flag" = true ]; then
    echo "Training modified U-Net model..."
    if [ "$use_hf_flag" = true ]; then
        echo "Using Hugging Face Transformers library"
        if [ -z "$max_len" ]; then
            python mod_train_unet.py --challenge "$challenge" --data_path "$data_path" --mask_type "$mask_type" --repo_id "$repo_id" --max_epochs "$max_epochs"
        else
            echo "Using a max_len of $max_len"
            python mod_train_unet.py --challenge "$challenge" --data_path "$data_path" --mask_type "$mask_type" --repo_id "$repo_id" --max_len "$max_len" --max_epochs "$max_epochs"
        fi
       
    else
        python mod_train_unet.py --challenge "$challenge" --data_path "$data_path" --mask_type "$mask_type" --max_epochs "$max_epochs"
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Training modified model failed."
        exit 1
    fi
fi

# Test modified model
if [ "$test_mod_flag" = true ]; then
    echo "Testing modified U-Net model..."
    python mod_train_unet.py --mode test --challenge "$challenge" --data_path "$data_path" 
    if [ $? -ne 0 ]; then
        echo "Error: Testing modified model failed."
        exit 1
    fi
fi

# Evaluate modified model
if [ "$eval_mod_model" = true ]; then
    echo "Evaluating modified U-Net model..."
    python evaluate.py --target-path "$data_path/ground_truth" --predictions-path "$output_path/reconstructions" --challenge "$challenge"
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation of modified model failed."
        exit 1
    fi
fi

# Run pretrained inference
if [ "$pretrain_inference_flag" = true ]; then
    echo "Running inference on pretrained U-Net model..."
    # Check to make sure the output directory exists
    if [ "$use_hf_flag" = true ]; then
        echo "Using Hugging Face Transformers library"
        if [ -z "$max_len" ]; then
            python run_pretrained_unet_inference.py --data_path "$data_path" --output_path "$output_path_pretrain" --challenge "$pretrain_challenge" --repo_id "$repo_id"
        else
            echo "Using a max_len of $max_len"
            python run_pretrained_unet_inference.py --data_path "$data_path" --output_path "$output_path_pretrain" --challenge "$pretrain_challenge" --repo_id "$repo_id" --max_len "$max_len"
        fi
        
    else
        python run_pretrained_unet_inference.py --data_path "$data_path" --output_path "$output_path_pretrain" --challenge "$pretrain_challenge"
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Pretrained inference failed."
        exit 1
    fi
fi

# Evaluate pretrained model
if [ "$eval_pretrained_model" = true ]; then
    echo "Evaluating pretrained U-Net model..."
    cd ../
    python fastmri/evaluate.py --target-path "$data_path/ground_truth" --predictions-path "$output_path/reconstructions" --challenge "$challenge"
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation of pretrained model failed."
        exit 1
    fi
fi

echo "All requested operations completed successfully."
