import numpy as np
import tensorflow as tf
from utils.text_connector.other import Graph
from utils.text_connector.text_connect_cfg import Config as TextLineCfg


def get_successions(text_proposals, heights, im_size, boxes_table, index):
    box = text_proposals[index]
    results = []
    for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, im_size[1])):
        adj_box_indices = boxes_table[left]
        for adj_box_index in adj_box_indices:
            if meet_v_iou(text_proposals, heights, adj_box_index, index):
                results.append(adj_box_index)
        if len(results) != 0:
            return results
    return results


def get_precursors(text_proposals, heights, boxes_table, index):
    box = text_proposals[index]
    results = []
    for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
        adj_box_indices = boxes_table[left]
        for adj_box_index in adj_box_indices:
            if meet_v_iou(text_proposals, heights, adj_box_index, index):
                results.append(adj_box_index)
        if len(results) != 0:
            return results
    return results


def is_succession_node(text_proposals, boxes_table, scores, index, succession_index):
    precursors = get_precursors(text_proposals, boxes_table, succession_index)
    if scores[index] >= np.max(scores[precursors]):
        return True
    return False


def meet_v_iou(text_proposals, heights, index1, index2):
    def overlaps_v(index1, index2):
        h1 = heights[index1]
        h2 = heights[index2]
        y0 = max(text_proposals[index2][1], text_proposals[index1][1])
        y1 = min(text_proposals[index2][3], text_proposals[index1][3])
        return max(0, y1 - y0 + 1) / min(h1, h2)

    def size_similarity(index1, index2):
        h1 = heights[index1]
        h2 = heights[index2]
        return min(h1, h2) / max(h1, h2)

    return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
           size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM


def build_graph(text_proposals, scores, im_size):
    heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

    boxes_table = [[] for _ in range(im_size[1])]
    for index, box in enumerate(text_proposals):
        boxes_table[int(box[0])].append(index)

    graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

    for index, box in enumerate(text_proposals):
        successions = get_successions(text_proposals, heights, im_size, boxes_table, index)
        if len(successions) == 0:
            continue
        succession_index = successions[np.argmax(scores[successions])]
        if is_succession_node(text_proposals, boxes_table, scores, index, succession_index):
            # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
            # have equal scores.
            graph[index, succession_index] = True

    sub_graphs = []
    for i in range(graph.shape[0]):
        if not graph[:, i].any() and graph[i, :].any():
            v = i
            sub_graphs.append([v])
            while graph[v, :].any():
                v = np.where(graph[v, :])[0][0]
                sub_graphs[-1].append(v)

    return sub_graphs
