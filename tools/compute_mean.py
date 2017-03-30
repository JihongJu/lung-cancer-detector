from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import (
    NPYDataLoader)

import yaml
with open("../tests/init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def increment_mean(avg_prev, n_prev, batch):
    """Estimate mean of arrays with size
    bigger than memory limits ONLINE by small batches.
    # Arguments
        avg_prev: Incremental average computed by the values from the previous batches
        n_prev: Number of values used previously
        batch: Batch of values to update the mean, an array or a list
    # Returns
        avg_cur: Incremental average updated with the current batch
        n_cur: Number of values used for computing the current mean
    """
    batch = np.array(batch)
    avg_batch = np.mean(batch)
    n_batch = batch.size
    if not n_prev:
        return avg_batch, n_batch
    n_cur = n_prev + n_batch
    avg_inc = (n_batch / float(n_cur)) * (avg_batch - avg_prev)
    avg_cur = avg_prev + avg_inc
    return avg_cur, n_cur


vol_data_gen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['test'])

train_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['trainval'])
test_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['test'])


mean = None
n = None


train_generator = vol_data_gen.flow_from_loader(
        volume_image_data_loader=train_vol_loader,
        batch_size=1, shuffle=True, seed=42)

for i in range(1396):
    print("Processing {}-th image (train): ({}, {})".format(i, mean, n))
    batch_x, _ = train_generator.next()
    mean, n = increment_mean(mean, n, batch_x)
    print("Current mean: {}".format(mean))


test_generator = vol_data_gen.flow_from_loader(
        volume_image_data_loader=test_vol_loader,
        batch_size=1, shuffle=True, seed=42)

for i in range(198):
    print("Processing {}-th image (test): ({}, {})".format(i, mean, n))
    batch_x, _ = test_generator.next()
    mean, n = increment_mean(mean, n, batch_x)
    print("Current mean: {}".format(mean))

print("Finished.\nFinal mean: {}".format(mean))


