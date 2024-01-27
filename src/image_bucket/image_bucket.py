# Released under MIT license
# Copyright (c) 2022 finetuneanon/zhifeng.ding (NovelAI/Anlatan LLC, vivo)
import math
import json
import random
import numpy as np
from dataclasses import dataclass
from loguru import logger
from typing import List


@dataclass
class BuckNode:
    width: int = -1  # 图片的宽度
    height: int = -1  # 图片的高度
    idx: int = -1  # 图片的索引

@dataclass
class BuckBatctIndex:
    batch_index: int = -1   # 一个桶里边批索引
    bucket_index: int = -1  # 全局桶索引
    batch_size: int = -1    # 批大小


class ImageBuckets:
    """构建多个空桶，然后将图片数据（不包含像素）放入桶中"""

    def __init__(self, max_size=(768, 512), divisible=64, step_size=8, min_dim=256, base_res=(512, 512), max_ar_error=4,
                 dim_limit=1024, seed=42):
        self.max_size = max_size  # 最大分辨率大小
        self.f = step_size  # 图像中分块的尺寸
        self.max_tokens = (max_size[0] / self.f) * (max_size[1] / self.f)  # 图像中块的数量限制
        self.div = divisible  # 分辨率按照div递增
        self.min_dim = min_dim  # 最小图像宽度
        self.dim_limit = dim_limit  # 最大宽度的限制
        self.base_res = base_res  # 基本分辨率大小
        self.max_ar_error = max_ar_error

        # 产生多个空桶
        self.resolutions, self.aspects = self.__gen_blank_bucket()
        self.buckets = {}
        self.buck_idxs: List[BuckBatctIndex] = [] 
        self.batch_count = 0
        self.seed = seed

    def __gen_blank_bucket(self):
        resolutions = []
        aspects = []

        w = self.min_dim
        while (w / self.f) * (self.min_dim / self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            # 循环是否中包含base_res
            got_base = False
            # 找到最大的h，在条件：h < dim_limit 和 块总数 < max_tokens
            while (w / self.f) * ((h + self.div) / self.f) <= self.max_tokens and (h + self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w) / float(h))
            w += self.div

        h = self.min_dim
        while (h / self.f) * (self.min_dim / self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            # 循环中是否跨越base_res
            got_base = False

            # 找到最大的w，在条件：w < dim_limit 和 块总数 < max_tokens
            while (h / self.f) * ((w + self.div) / self.f) <= self.max_tokens and (w + self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w) / float(h))
            h += self.div

        # 尺寸（宽，高）对应的宽高比
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        aspects = np.array(list(map(lambda x: res_map[x], resolutions)))
        resolutions = np.array(resolutions)
        # print(res_map)
        return resolutions, aspects

    def _get_bid(self, info: BuckNode):
        """ get bucket id by image aspect

        Args:
            info (BuckNode): a node of image

        Returns:
            tuple: bucket id and aspect error
        """
        aspect = float(info.width) / float(info.height)
        bucket_id = np.abs(self.aspects - aspect).argmin()
        error = abs(self.aspects[bucket_id] - aspect)
        return bucket_id, error
    
    def clear(self):
        self.buckets = {}
    
    def inject(self, node: BuckNode):
        """ insert node into bucket

        Args:
            node (BuckNode): a node of image

        Returns:
            float: image aspect error(-1: error > max_ar_error)
        """
        post_id = node.idx
        bucket_id, error = self._get_bid(node)

        self.buckets[bucket_id] = self.buckets.get(bucket_id, [])
        if error < self.max_ar_error:
            self.buckets[bucket_id].append(post_id)
            return error
        
        return -1

    def inject_stream(self, nodes: List[BuckNode]):
        aspect_errors = []
        skipped = 0
        for node in nodes:
            post_id = node.idx
            bucket_id, error = self._get_bid(node)
            self.buckets[bucket_id] = self.buckets.get(bucket_id, [])
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                aspect_errors.append(error)
            else:
                skipped += 1

        aspect_errors = np.array(aspect_errors)
        logger.info(f"skipped images: {skipped}")
        logger.info(f"aspect error: mean {aspect_errors.mean()}, median {np.median(aspect_errors)}, max {aspect_errors.max()}")

        return skipped != len(nodes)  

    def show(self):
        logger.info(json.dumps(self.resolutions.tolist()))
        logger.info(json.dumps(self.aspects.tolist()))

        for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
            logger.info(
                f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, \
                  entries {len(self.buckets[bucket_id])}")

    def make(self, batch_size, shuffle=True):
        """ make a batch node

        Args:
            batch_size (int): a batch size of image
            shuffle (bool, optional): shuffle image or not. Defaults to True.
        """
        buck_idxs = []
        for bid in self.buckets.keys():
            batch_count = int(math.ceil(len(self.buckets[bid]) / batch_size))
            for batch_index in range(batch_count):
                buck_idxs.append(BuckBatctIndex(batch_index, bid, batch_size))
        
        self.buck_idxs = buck_idxs
        self.batch_count = len(buck_idxs)
        if shuffle:
            self.shuffle(epoch=0)
    
    def __len__(self):
        return self.batch_count
    
    def transforms(self, bidxs: List[int], bucket, resolution):
        return bidxs
    
    def __getitem__(self, idx):
        bucket = self.buckets[self.buck_idxs[idx].bucket_index]
        batch_size = self.buck_idxs[idx].batch_size
        image_index = self.buck_idxs[idx].batch_index * batch_size 
        resolution = self.resolutions[self.buck_idxs[idx].bucket_index]
        return self.transforms(bucket[image_index:image_index + batch_size], bucket, resolution)
    
    def shuffle(self, epoch):
        random.seed(self.seed+epoch)
        for key in self.buckets.keys():
            random.shuffle(self.buckets[key])
        random.shuffle(self.buck_idxs)
