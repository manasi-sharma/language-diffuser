# language-diffuser

Installation instructions for environment:
Install dependencies from decision diffuser (https://github.com/anuragajay/decision-diffuser/tree/main/code) and CALVIN (https://github.com/mees/calvin)

## DECISION DIFFUSER:
1. in environment.yml, comment out 'ml_logger==0.8.69'
2. Run:
conda env create -f environment.yml
conda activate decdiff
3. conda install -c conda-forge pycurl
4. pip install typed-argument-parser
5. numpy == 1.23.5

## CALVIN:
6. cd into the CALVIN directory
7. run sh install.sh
8. pip install torchmetrics==v0.6.0
9. pip install params-proto waterbear

# commands to run langage diffuser:
[interactive mode]
python scripts/train.py datamodule.root_data_dir=/iliad/u/manasis/language-diffuser/code/calvin_debug_dataset datamodule/datasets=vision_lang

[sbatch]
python simple_sbatch.py --entry-point scripts/train.py --arguments datamodule.root_data_dir=/iliad/u/manasis/language-diffuser/code/calvin_debug_dataset datamodule/datasets=vision_lang  --account iliad --partition iliad --nodelist iliad4 --cpus 2 --gpus 1 --mem 20G --job-name new_langdiffuser_1e6_10000_200_bsize_32
