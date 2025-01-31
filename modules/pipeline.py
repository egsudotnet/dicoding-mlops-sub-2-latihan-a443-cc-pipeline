import os
import sys
from typing import Text
 
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
 
PIPELINE_NAME = "egihsugiatna-pipeline"
 
# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/stress_transform.py"
TRAINER_MODULE_FILE = "modules/stress_trainer.py"
# requirement_file = os.path.join(root, "requirements.txt")
 
# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")