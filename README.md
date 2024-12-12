CS7643 Final Project
 
1. Description
 
This branch includes the code for our Final Project: https://github.com/travisdoesmath/fastMRI.
 
Modified U-Net Model: The Modified U-Net model is an enhanced version of the original U-Net. It integrates a weighted loss function that upweights errors within the region of interest, encouraging the network to prioritize reconstruction quality at the center of the MRI scan.
 
Weighted U-Net Model: The Weighted U-Net model incorporates annotation-based weighting into the training process. This model focuses on improving reconstruction accuracy in diagnostically critical areas by emphasizing regions of interest (e.g., annotated abnormalities in MRI scans) with higher weights.
 
Knowledge Distillation (KDist) Model: The KDist model leverages a teacher-student framework, enabling the student to learn from the teacher's predictions and achieve high-quality reconstructions.

-------------------------------------------------------------------------------
 
2. Installation and Execution
 
Prerequisites: Python 3.10 or higher.
This can be achieved easily with Homebrew: brew install python@3.10

Commands to run in order from a Mac terminal.
 
cd ~/Desktop
mkdir Project
cd Project
python3.10 -m venv venv
source venv/bin/activate (venv\Scripts\activate if using Windows)
git clone https://github.com/travisdoesmath/fastMRI.git
cd fastMRI
pip install -r requirements.txt (Remove the following before running: -e git+https://github.com/travisdoesmath/fastMRI.git@a6e62887245bfa9de1c4852ba3b52a1a4b6f5ae2#egg=fastmri)
pip install -e .

-------------------------------------------------------------------------------
 
3. Running the Scripts (Train, Test, and Evaluate)
 
cd fastmri_examples/mod_unet/
 
Training the Modified Model: bash runner.sh -use_hf -train-mod --max_len 4 --max_epochs 1
If you encounter errors, execute these commands:
pip install huggingface-hub==0.26.3
pip uninstall numpy
pip install "numpy<2.0"              ---> Downgrades to a compatible version; numpy-1.26.4)
pip install tensorboard              ---> Required for logging
 
Testing the Modified Model: bash runner.sh -use_hf -test-mod --max_len 4   
If you encounter errors, execute these commands:
pip uninstall pytorch-lightning
pip install pytorch-lightning==1.9.0 --->Downgrades to a compatible version
 
Note: If Testing the Modified Model again make sure to delete the reconstructions folder.
Path: Desktop/Project/fastMRI/fastmri_examples/mod_unet/output_mod/reconstructions
rm -r output_mod/reconstructions
 
Evaluating the Modified Model: bash runner.sh -eval-mod -use_hf --max_len 4   
If you encounter errors, execute these commands:
N/A
 
Additional Examples:

Training the KDist Model: bash runner.sh -use_hf -train-kdist --max_len 4 --max_epochs 1
Testing the KDist Model: bash runner.sh -use_hf -test-mod --max_len 4
Evaluating the KDist Model: bash runner.sh -eval-mod -use_hf --max_len 4

Training the Weighted Model: bash runner.sh -use_hf -train-weighted --max_len 4 --max_epochs 1
Testing the Weighted Model: bash runner.sh -use_hf -test-mod --max_len 4
Evaluating Weighted Model: bash runner.sh -eval-mod -use_hf --max_len 4
 
