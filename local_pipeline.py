# 1. Pada bagian awal tentunya kita perlu memanggil berbagai library dan variabel yang dibutuhkan seperti berikut.
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

# 2. Membuat sebuah fungsi bernama init_local_pipeline(). Fungsi inilah yang digunakan untuk menyatukan seluruh TFX component. Untuk menjalankan tugasnya, ia membutuhkan dua buah parameter input, yaitu components dan pipeline_root.
def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available 
        # during execution time.
        "----direct_num_workers=0" 
    ]
    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )

# 3. menjalankan pipeline
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    
    from modules.components import init_components
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )
    
    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)


# conda create -n stress python=3.9.12
# conda activate stress
# uv pip install jupyter scikit-learn tensorflow tfx==1.11 flask joblib
# uv pip install pip setuptools==70
# cd "dicoding-mlops-sub-2\dicoding-mlops-sub-2"


# 4. Untuk menjalankan berkas local_pipeline.py, tulislah perintah berikut pada terminal atau windows PowerShell.
# python .\local_pipeline.py 


# cd "dicoding-mlops-sub-2"
# cd latihan-dicoding
# cd a443-es-pipeline
# cd "dicoding-mlops-sub-2/latihan-dicoding/a443-es-pipeline"



# cd "\Programming\dicoding\Machine Learning\mlops\dicoding-mlops-sub-2\latihan-dicoding\a443-es-pipeline"


# Poer shell heroku create es-prediction
# Conda jalankan env
 
# 1. Install Python 3.9.21
# uv python install 3.9.21
# 2. Buat Virtual Env
# uv venv --python 3.9.21
# 3. Aktifin env
# .venv\Scripts\activate
# 4. Install library 
# uv pip install jupyter scikit-learn tensorflow tfx==1.11 flask joblib
# 5. Install pip & turunin setuptools
# uv pip install pip setuptools==70

# conda create -n stress3 python=3.9.21
# conda activate stress3
# uv pip install jupyter scikit-learn tensorflow tfx==1.11 flask joblib
# # # # pip install -r requirements.txt

# cd "dicoding-mlops-sub-2\dicoding-mlops-sub-2"

# -================================
# heroku container:login 
# heroku stack:set container -a es-prediction
# heroku container:push web -a es-prediction
# heroku container:release web -a es-prediction


# heroku container:login
# heroku container:push web -a es-prediction
# heroku container:release web -a es-prediction

# =====================================
# heroku login
# heroku create es-prediction 
# heroku container:login

# build image
# heroku stack:set container -a es-prediction
# docker push registry.heroku.com/es-prediction/web


# heroku container:push web -a es-prediction
# heroku container:release web -a es-prediction


# docker tag a443ccpipeline:latest registry.heroku.com/es-prediction/web


# localhost:8501/v1/models/saved_model/metadata

# docker build -t cc-monitoring .\monitoring\
# docker run -p 9090:9090 cc-monitoring  
# http://localhost:9090/

# bool xla_cpu_compilation_enabled = true;
# bool xla_gpu_compilation_enabled = true;



# =====================================
# =====================================
# heroku login
# heroku create es-stress 
# heroku container:login

# # # build image
# heroku stack:set container -a es-stress 


# heroku container:push web -a es-stress
# heroku container:release web -a es-stress


# docker tag a443ccpipeline:latest registry.heroku.com/es-stress/web


# localhost:8501/v1/models/saved_model/metadata

# docker build -t cc-monitoring .\monitoring\
# docker run -p 9090:9090 cc-monitoring  
# http://localhost:9090/

# bool xla_cpu_compilation_enabled = true;
# bool xla_gpu_compilation_enabled = true;

# https://es-stress-9bb91864c40c.herokuapp.com/v1/models/stress-model/metadata

 
# web: tensorflow_model_server --port=8500 --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} --xla_cpu_compilation_enabled=false --xla_gpu_compilation_enabled=false --num_load_threads=1 --num_unload_threads=1 --tensorflow_session_parallelism=1

