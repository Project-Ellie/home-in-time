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

from prep.prep_tools import join_paths
from prep.run_job import run_job

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--project',
        help = 'GCP project ID',
        type = str,
        required = True
    )
    parser.add_argument(
        '--base_dir',
        help = 'base_dir from which to interpret all other relative paths',
        required = True
    )
    parser.add_argument(
        '--prefix',
        help = 'Prefix for data file names',
        type = str,
        required = True
    )

    parser.add_argument(
        '--metadata_dir',
        help = 'directory to write metadata to',
        type = str,
        default = 'metadata'
    )
    parser.add_argument(
        '--tmp_dir',
        help = 'directory to write temporary data to',
        type = str,
        default = 'tmp'
    )
    parser.add_argument(
        '--stage_dir',
        help = 'directory to write staging data to',
        type = str,
        default = 'staging'
    )
    
    parser.add_argument(
        '--train_dir',
        help = 'directory to write training data to',
        type = str,
        default = 'train_data'
    )
    parser.add_argument(
        '--eval_dir',
        help = 'directory to write evaluation data to',
        type = str,
        default = 'eval_data'
    )
    parser.add_argument(
        '--test_dir',
        help = 'directory to write test data to',
        type = str,
        default = 'test_data'
    )

    parser.add_argument(
        '--encode',
        help = 'csv, tfrecord or both, defaults to tfrecord',
        type = str,
        default = 'tfrecord')
    
    parser.add_argument(
        '--fractions',
        help = 'data split: comma-seperated fractions of 100, like e.g 80,10,10',
        type = str,
        default = '80,10,10'
    )
    parser.add_argument(
        '--sample_rate',
        help = 'sample rate. Fraction of records in total. 0.1 would mean 10 percent',
        type = float,
        default = 1.0
    )
    parser.add_argument(
        '--max_workers',
        help = 'max number of workers to be spawned if using DataFlowRunner',
        default = 24
    )

    parser.add_argument(
        '--runner',
        help = 'DataFlowRunner or DirectRunner',
        default = 'DirectRunner'
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # Run the preprocessing job:
    try:
        run_job(join_paths(arguments))
    except:
        traceback.print_exc()