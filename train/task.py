# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import traceback
import argparse
import json
import os
import tensorflow as tf

from train.train_tools import join_paths
from train.train_and_evaluate import train_and_evaluate

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File specifications
    parser.add_argument('--base_dir', required = True,
        help = 'base_dir from which to interpret all other relative paths')
    
    parser.add_argument('--metadata_dir',required=True,
        help = 'base_dir from which to interpret all other relative paths')
    
    parser.add_argument('--train_data_pattern',required = True,
        help = 'GCS path glob pattern to training data')
    
    parser.add_argument('--eval_data_pattern', required=True,
        help = 'GCS patch glob pattern to evaluation data')
    
    parser.add_argument('--file_format', required=True,
        help='File format. One of "csv", "tfr"')
    
    parser.add_argument('--model_dir', required=True,
        help = 'GCS location to write checkpoints and export models')

    
    # Hyperparameters
    parser.add_argument('--train_batch_size', default=512, type=int,
        help = 'Batch size for training steps')
    
    parser.add_argument('--max_train_steps', default=5000, type=int,
        help = 'Max num steps to run the training job up to')
    
    parser.add_argument('--learning_rate', default=1e-4, type=float,
        help = 'How long to wait before running first evaluation')

    parser.add_argument( '--nbuckets', default=10, type=int,
        help = 'NOT USED. Number of buckets into which to discretize lats and lons')

    parser.add_argument('--hidden_units', default = '128 32 4', 
        help = 'NOT USED: Hidden layer sizes to use for deep networks')

    parser.add_argument('--optimizer', default = 'sgd',
        help = 'Optimizer to be used, one of "sgd", "adam", "adagrad"')
    
    # Execution parameters
    parser.add_argument('--eval_batch_size', default=512, type=int,
        help = 'Batch size for evaluation steps')
    
    parser.add_argument('--eval_steps', default=10, type=int,
        help = 'Number of steps to run evalution for at each checkpoint')
    
    parser.add_argument('--throttle_secs', default=30, type=int,
        help = 'number of seconds between two evaluations')

    parser.add_argument('--distribute', type=bool,
        help = 'boolean. Whether or not to distribute between GPUs')
    
    parser.add_argument('--prefetch_buffer_size', default=1024, type=int,
        help = 'how many batches to pre-fetch into GPU memory')
    
    parser.add_argument('--log_step_count_steps', default=1000, type=int,
        help="log current training loss every this number of steps")
    
    parser.add_argument('--save_summary_steps', default=100, type=int,
        help="save summaries every after this number of steps")
    
    parser.add_argument('--hypothesis', required=True, type=str,
        help="name of the hypothesis. Examine 'hypotheses.py' for a list of available hypotheses.")
    
    parser.add_argument('--save_checkpoints_steps', default=2000, type=int,
        help="number of steps to go until another check_point is created.")        
    
    parser.add_argument('--parser_num_threads', default=16, type=int,
        help="Number of Parser threads to use")
    
    parser.add_argument('--reader_num_threads', default=16, type=int,
        help="Number of io reader threads to use")
    
    parser.add_argument('--shuffle_buffer_size', default=10000, type=int,
        help="Shuffle buffer size")
    
    parser.add_argument('--sloppy_ordering', default=True, type=bool,
        help="Whether to allow non-deterministic ordering. Boosts performance")
    
    parser.add_argument('--job-dir', default='whatever',
        help = 'this model ignores this field, but it is required by gcloud')


    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    arguments['model_dir'] = os.path.join(
        arguments['model_dir'],
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    ) 

    if arguments['distribute']:
        print("###############################################################")
        print("       Running distributed!")
        print("###############################################################")

    # Run the training job:
    try:
        train_and_evaluate(join_paths(arguments))
    except:
        traceback.print_exc()