# File   : bucket.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import math
import logging
import numpy as np
from random import shuffle
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BySequenceLengthSampler(Sampler):

    def __init__(self, dataset, bucket_boundaries, bucket_batch_size=[], rm_the_long_sents=True,
                 audo_set_batch_size=True, max_frames_one_batch=20000, drop_last=False, short_first=False):

        assert isinstance(bucket_boundaries, list)
        assert isinstance(bucket_batch_size, list)

        self.index_length_pair = dataset.index_length_pair()

        self.drop_last = drop_last
        self.bucket_boundaries = bucket_boundaries    # the number of frames [100, 200, 300, 400]
        self.bucket_batch_size = bucket_batch_size
        self.audo_set_batch_size = audo_set_batch_size
        self.max_frames_one_batch = max_frames_one_batch
        self.short_first = False
        if self.short_first:
            logger.info('[Sampler] Apply Short-Utterence-First!')

        if self.audo_set_batch_size:
            assert self.max_frames_one_batch > 0
            logger.info('[Sampler] Auto Set batch_size based on max_frames_one_batch (%d).' % int(self.max_frames_one_batch))

        self.rm_the_long_sents = rm_the_long_sents  # remove the longer sentencer over the max length of buckets
                                                    # if False, it will build a new bucket to store the long sentences
        self.max_length = self.bucket_boundaries[-1] if self.rm_the_long_sents else np.iinfo(np.int32).max

        # build buckets
        self.num_of_buckets = len(self.bucket_boundaries) if self.rm_the_long_sents else len(self.bucket_boundaries) + 1
        self.buckets = {}
        for bucket_id in range(self.num_of_buckets):
            boundary_min = 0 if bucket_id == 0 else self.bucket_boundaries[bucket_id - 1]
            boundary_max = np.iinfo(np.int32).max if bucket_id == len(self.bucket_boundaries) else self.bucket_boundaries[bucket_id]
            batch_size = 0 if self.audo_set_batch_size else self.bucket_batch_size[min(bucket_id, len(self.bucket_batch_size)-1)]
            self.buckets[str(bucket_id)] = {
                'index_length_pair': [],
                'batch_size': batch_size,
                'boundary': [boundary_min, boundary_max]
            }

        self.batch_list = []

        self.put_data_pair_into_buckets()
        self.split_the_bucket_into_batch()

    def __iter__(self):
        if not self.short_first:
            shuffle(self.batch_list)

        # self.split_the_bucket_into_batch()

        for (bucket_id, index_list) in self.batch_list:
            logger.debug('[Sampler] Read %d utterances from bucket %d.' % (len(index_list), bucket_id))
            yield index_list

    def __len__(self):
        logger.info('The number of batch %d' % len(self.batch_list))
        return len(self.batch_list)

    def put_data_pair_into_buckets(self):

        # put index_length_pair into the buckets
        rm_conut = 0
        for index, length in self.index_length_pair:
            if self.rm_the_long_sents:
                if length > self.max_length:
                    rm_conut += 1
                    continue
            bucket_id = self.element_to_bucket_id(length)
            self.buckets[str(bucket_id)]['index_length_pair'].append((index, length))
        logger.info('[Sampler] Delete %d utterances over the maximum bounday!' % rm_conut)
        logger.info('[Sampler] Put the data pair into the different buckets!')

    def split_the_bucket_into_batch(self):
        # split the bucket into batch

        self.batch_list = []

        for bucket_id in self.buckets.keys():
            boundary = self.buckets[bucket_id]['boundary']
            if len(self.buckets[bucket_id]['index_length_pair']) == 0:
                logger.debug('[Sampler] Skip the No.%s bucket with boundary from %d to %d' % (bucket_id, boundary[0], boundary[1]))
                continue

            shuffle(self.buckets[bucket_id]['index_length_pair'])
            logger.debug('[Sampler] There are %d utterances in the No.%s bucket (from %d to %d)' % 
                        (len(self.buckets[bucket_id]['index_length_pair']), bucket_id, boundary[0], boundary[1]))

            if self.audo_set_batch_size:
                batch_list = self.generate_batches_based_length(bucket_id, self.buckets[bucket_id]['index_length_pair'])
            else:
                batch_list = self.generate_batches(bucket_id,
                    self.buckets[bucket_id]['index_length_pair'],
                    self.buckets[bucket_id]['batch_size'])
            self.batch_list.extend(batch_list)
            logger.debug('[Bucket Sampler] There are %d batches in the No.%s bucket.' % (len(batch_list), bucket_id))

    def generate_batches(self, bucket_id, index_lenth_pair, batch_size):

        batch_list = []
        num_pairs = len(index_lenth_pair)
        num_batch = math.floor(num_pairs / batch_size) if self.drop_last else math.ceil(num_pairs / batch_size)

        if self.short_first:
            index_lenth_pair = sorted(index_lenth_pair, key=lambda x: x[1], reverse=False)

        for i in range(int(num_batch)):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, num_pairs)
            index_list = [index for index, _ in index_lenth_pair[start_idx : end_idx]]
            batch_list.append((int(bucket_id), index_list))

        if self.drop_last:
            logger.debug('[Bucket Sampler] Drop %d uttterances in the No.%s bucket.' % (num_pairs - num_batch * batch_size, bucket_id))
        
        return batch_list

    def generate_batches_based_length(self, bucket_id, index_lenth_pair):

        batch_list = []
        index_list = []

        if self.short_first:
            index_lenth_pair = sorted(index_lenth_pair, key=lambda x: x[1], reverse=False)

        accu_length = 0
        for index, length in index_lenth_pair:
            if accu_length + length > self.max_frames_one_batch:
                batch_list.append((int(bucket_id), index_list))
                accu_length = 0
                index_list = []
            
            index_list.append(index)
            accu_length += length

        if len(index_list) > 0 and not self.drop_last:
            batch_list.append((int(bucket_id), index_list))
        else:
            logger.debug('[Bucket Sampler] Drop %d uttterances in the No.%s bucket.' % (len(index_list), bucket_id))
        
        return batch_list

    def element_to_bucket_id(self, seq_length):
        buckets_min = [0] + self.bucket_boundaries
        buckets_max = self.bucket_boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.greater(seq_length, buckets_min),
            np.less_equal(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

    def shuffle_batch_in_bucket(self):
        if self.short_first:
            logger.info('[Bucket Sampler] Apply Short First and Do not shuffle the utterences!')
        else:
            self.split_the_bucket_into_batch()
            logger.info('[Bucket Sampler] Shuffle the utterances in a bucket!')
