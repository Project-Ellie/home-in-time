import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

##################################################################################
# Specifying signature data
##################################################################################
SIGNATURE_STR_COLUMNS = ['ARR', 'AIRLINE']
SIGNATURE_INT_COLUMNS = ['DEP_DOW', 'DEP_T']
SIGNATURE_FLOAT_COLUMNS = ['DEP_LAT', 'DEP_LON', 'DEP_DELAY', 'MEAN_TEMP_DEP', 
                 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'ARR_LAT', 
                 'ARR_LON', 'ARR_DELAY', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR']

SIGNATURE_COLUMNS=(SIGNATURE_INT_COLUMNS + 
                   SIGNATURE_FLOAT_COLUMNS + 
                   SIGNATURE_STR_COLUMNS)

SIGNATURE_SCHEMA={}
for t, cols in [
    (tf.float32, SIGNATURE_FLOAT_COLUMNS), 
    (tf.int64, SIGNATURE_INT_COLUMNS),
    (tf.string, SIGNATURE_STR_COLUMNS)]:
    SIGNATURE_SCHEMA.update({
        col : dataset_schema.ColumnSchema(
            t, [], dataset_schema.FixedColumnRepresentation())
                   for col in cols})

SIGNATURE_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.Schema(SIGNATURE_SCHEMA))


##################################################################################
# Specifying training data
##################################################################################
TRAINING_INT_COLUMNS=['DEP_DOW', 'DEP_HOD', 'AIRLINE', 'ARR']
TRAINING_FLOAT_COLUMNS=[
    'DEP_LAT', 'DEP_LON', 'MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 
    'DEP_DELAY', 'ARR_LAT', 'ARR_LON', 'ARR_DELAY',
    'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR',  
    'DIFF_LAT', 'DIFF_LON', 'DISTANCE']

TRAINING_COLUMNS=(TRAINING_INT_COLUMNS + 
                  TRAINING_FLOAT_COLUMNS)

TRAINING_SCHEMA={}
for t, cols in [(tf.float32, TRAINING_FLOAT_COLUMNS), 
                (tf.int64, TRAINING_INT_COLUMNS)]:
    TRAINING_SCHEMA.update({
        col : dataset_schema.ColumnSchema(t, [], dataset_schema.FixedColumnRepresentation())
                   for col in cols})
TRAINING_METADATA = dataset_metadata.DatasetMetadata(dataset_schema.Schema(TRAINING_SCHEMA))

TRAINING_DEFAULTS = [
    [0], [0], [0], [0], 

    [0.], [0.], 
    [0.], [0.], [0.], 
    [0.], 

    [0.], [0.], [0.],
    [0.], [0.], [0.], 
    
    [0.], [0.], [0.]]

# Need strictly increasing column names for tf.data csv decoder
from operator import itemgetter
C_D = zip(TRAINING_COLUMNS, TRAINING_DEFAULTS)
C_D.sort(key=itemgetter(0))


ORDERED_TRAINING_COLUMNS = [item[0] for item in C_D]
ORDERED_TRAINING_DEFAULTS = [item[1] for item in C_D]

LABEL_COLUMN = 'ARR_DELAY'
