# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Sergey Karayev
# Modified by Yuwen Xiong, from from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

strategy = 2  

def bbox_overlaps_cython(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_overlaps_centerIns_cython(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] centerin_overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        query_box_center=[(query_boxes[k, 2] + query_boxes[k, 0])/2,(query_boxes[k, 3] + query_boxes[k, 1])/2]
        for n in range(N):
            if boxes[n,4]==query_boxes[k,4]
                iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                    )
                    if ih > 0:
                        ua = float(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
                        if strategy==1:
                            if query_box_center[0]<boxes[n,2] and query_box_center[0]>boxes[n,0]:
                                if query_box_center[1]<boxes[n,3] and query_box_center[1]>boxes[n,1]:
                                    centerin_overlaps[n, k] = overlaps[n, k]                
                        elif strategy==2:
                            box_center=[(boxes[n, 2] - boxes[n, 0])/2,(boxes[n, 3] - boxes[n, 1])/2]
                            if query_box_center[0]<boxes[n,2] and query_box_center[0]>boxes[n,0]:
                                if query_box_center[1]<boxes[n,3] and query_box_center[1]>boxes[n,1]:
                                    if box_center[0]<query_boxes[k,2] and box_center[0]>query_boxes[k,0]:
                                        if box_center[1]<query_boxes[k,3] and box_center[1]>query_boxes[k,1]:
                                            centerin_overlaps[n,k] = overlaps[n, k]                
    return overlaps,centerin_overlaps