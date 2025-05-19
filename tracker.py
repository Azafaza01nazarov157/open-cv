import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import pandas as pd
from datetime import datetime

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_name = None

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.analytics = pd.DataFrame(columns=['timestamp', 'track_id', 'class', 'x', 'y', 'w', 'h'])

    def update(self, dets, class_names):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if dets.size > 0:
            matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
            
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :])
                self.trackers[m[1]].class_name = class_names[m[0]]

            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.class_name = class_names[i]
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1], [trk.class_name])).reshape(1, -1))
                # Log analytics
                self._log_analytics(trk, d)
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def _log_analytics(self, trk, d):
        timestamp = datetime.now()
        x, y, w, h = d[0], d[1], d[2]-d[0], d[3]-d[1]
        self.analytics = pd.concat([self.analytics, pd.DataFrame({
            'timestamp': [timestamp],
            'track_id': [trk.id],
            'class': [trk.class_name],
            'x': [x],
            'y': [y],
            'w': [w],
            'h': [h]
        })], ignore_index=True)

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def iou(bb_det, bb_trk):
        xx1 = np.maximum(bb_det[0], bb_trk[0])
        yy1 = np.maximum(bb_det[1], bb_trk[1])
        xx2 = np.minimum(bb_det[2], bb_trk[2])
        yy2 = np.minimum(bb_det[3], bb_trk[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
                  + (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1]) - wh)
        return o 