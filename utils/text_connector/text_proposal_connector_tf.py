import numpy as np
import tensorflow as tf
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


def fit_y(X, Y, x1, x2):
    len(X) != 0
    # if X only include one point, the function will get line y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes0 = tf.maximum(tf.minimum(boxes[:, 0], im_shape[1] - 1), 0)
    boxes1 = tf.maximum(tf.minimum(boxes[:, 1], im_shape[0] - 1), 0)
    return tf.stack([boxes0, boxes1], axis=1)


def get_text_lines(text_proposals, scores, im_size):
    # tp=text proposal
    tp_groups = build_graph(text_proposals, scores, im_size)
    text_lines = np.zeros((len(tp_groups), 5), np.float32)

    for index, tp_indices in enumerate(tp_groups):
        text_line_boxes = text_proposals[list(tp_indices)]

        x0 = np.min(text_line_boxes[:, 0])
        x1 = np.max(text_line_boxes[:, 2])

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        # the score of a text line is the average score of the scores
        # of all text proposals contained in the text line
        score = scores[list(tp_indices)].sum() / float(len(tp_indices))

        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)
        text_lines[index, 4] = score

    text_lines = clip_boxes(text_lines, im_size)

    text_recs = np.zeros((len(text_lines), 9), np.float)
    index = 0
    for line in text_lines:
        xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
        text_recs[index, 0] = xmin
        text_recs[index, 1] = ymin
        text_recs[index, 2] = xmax
        text_recs[index, 3] = ymin
        text_recs[index, 4] = xmax
        text_recs[index, 5] = ymax
        text_recs[index, 6] = xmin
        text_recs[index, 7] = ymax
        text_recs[index, 8] = line[4]
        index = index + 1

    return text_recs