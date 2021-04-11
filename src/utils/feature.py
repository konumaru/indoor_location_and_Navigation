import re
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from pylab import imread
import matplotlib.pyplot as plt


def un_pickle(filename):
    with open(filename, "rb") as fo:
        p = pickle.load(fo)
    return p


def to_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, -1)


class FeatureStore:

    # necessayr to re-check
    floor_convert = {
        "1F": 0,
        "2F": 1,
        "3F": 2,
        "4F": 3,
        "5F": 4,
        "6F": 5,
        "7F": 6,
        "8F": 7,
        "9F": 8,
        "B": -1,
        "B1": -1,
        "B2": -2,
        "B3": -3,
        "BF": -1,
        "BM": -1,
        "F1": 0,
        "F2": 1,
        "F3": 2,
        "F4": 3,
        "F5": 4,
        "F6": 5,
        "F7": 6,
        "F8": 7,
        "F9": 8,
        "F10": 9,
        "L1": 0,
        "L2": 1,
        "L3": 2,
        "L4": 3,
        "L5": 4,
        "L6": 5,
        "L7": 6,
        "L8": 7,
        "L9": 8,
        "L10": 9,
        "L11": 10,
        "G": 0,
        "LG1": 0,
        "LG2": 1,
        "LM": 0,
        "M": 0,
        "P1": 0,
        "P2": 1,
    }

    df_types = [
        "accelerometer",
        "accelerometer_uncalibrated",
        "beacon",
        "gyroscope",
        "gyroscope_uncalibrated",
        "magnetic_field",
        "magnetic_field_uncalibrated",
        "rotation_vector",
        "waypoint",
        "wifi",
    ]

    # https://github.com/location-competition/indoor-location-competition-20
    df_type_cols = {
        "accelerometer": ["timestamp", "x", "y", "z", "accuracy"],
        "accelerometer_uncalibrated": [
            "timestamp",
            "x",
            "y",
            "z",
            "x2",
            "y2",
            "z2",
            "accuracy",
        ],
        "beacon": [
            "timestamp",
            "uuid",
            "major_id",
            "minor_id",
            "tx_power",
            "rssi",
            "distance",
            "mac_addr",
            "timestamp2",
        ],
        "gyroscope": ["timestamp", "x", "y", "z", "accuracy"],
        "gyroscope_uncalibrated": [
            "timestamp",
            "x",
            "y",
            "z",
            "x2",
            "y2",
            "z2",
            "accuracy",
        ],
        "magnetic_field": ["timestamp", "x", "y", "z", "accuracy"],
        "magnetic_field_uncalibrated": [
            "timestamp",
            "x",
            "y",
            "z",
            "x2",
            "y2",
            "z2",
            "accuracy",
        ],
        "rotation_vector": ["timestamp", "x", "y", "z", "accuracy"],
        "waypoint": ["timestamp", "x", "y"],
        "wifi": [
            "timestamp",
            "ssid",
            "bssid",
            "rssi",
            "frequency",
            "last_seen_timestamp",
        ],
    }

    dtype_dict = {}
    dtype_dict["accelerometer"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    }
    dtype_dict["accelerometer_uncalibrated"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "x2": float,
        "y2": float,
        "z2": float,
        "accuracy": int,
    }
    dtype_dict["beacon"] = {
        "timestamp": int,
        "uuid": str,
        "major_id": str,
        "minor_id": str,
        "tx_power": int,
        "rssi": int,
        "distance": float,
        "mac_addr": str,
        "timestamp2": int,
    }
    dtype_dict["gyroscope"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    }
    dtype_dict["gyroscope_uncalibrated"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "x2": float,
        "y2": float,
        "z2": float,
        "accuracy": int,
    }
    dtype_dict["magnetic_field"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    }
    dtype_dict["magnetic_field_uncalibrated"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "x2": float,
        "y2": float,
        "z2": float,
        "accuracy": int,
    }
    dtype_dict["rotation_vector"] = {
        "timestamp": int,
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    }
    dtype_dict["waypoint"] = {"timestamp": int, "x": float, "y": float, "z": float}
    dtype_dict["wifi"] = {
        "timestamp": int,
        "ssid": str,
        "bssid": str,
        "rssi": int,
        "frequency": int,
        "last_seen_timestamp": int,
    }

    def __init__(
        self,
        site_id,
        floor,
        path_id,
        input_path="../input/indoor-location-navigation/",
        save_path="../data/working",
    ):
        self.site_id = site_id.strip()
        self.floor = floor.strip()
        self.n_floor = self.floor_convert[self.floor]
        self.path_id = path_id.strip()

        self.input_path = input_path
        assert Path(input_path).exists(), f"input_path does not exist: {input_path}"

        self.save_path = save_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        self.site_info = SiteInfo(
            site_id=self.site_id, floor=self.floor, input_path=self.input_path
        )

    def _flatten(self, l):
        return list(itertools.chain.from_iterable(l))

    def multi_line_spliter(self, s):
        matches = re.finditer("TYPE_", s)
        matches_positions = [match.start() for match in matches]
        split_idx = (
            [0]
            + [matches_positions[i] - 14 for i in range(1, len(matches_positions))]
            + [len(s)]
        )
        return [s[split_idx[i] : split_idx[i + 1]] for i in range(len(split_idx) - 1)]

    def load_df(self):
        path = str(
            Path(self.input_path)
            / f"train/{self.site_id}/{self.floor}/{self.path_id}.txt"
        )
        with open(path) as f:
            data = f.readlines()

        modified_data = []
        for s in data:
            if s.count("TYPE_") > 1:
                lines = self.multi_line_spliter(s)
                modified_data.extend(lines)
            else:
                modified_data.append(s)
        del data
        self.meta_info_len = len([d for d in modified_data if d[0] == "#"])
        self.meta_info_df = pd.DataFrame(
            [
                m.replace("\n", "").split(":")
                for m in self._flatten(
                    [d.split("\t") for d in modified_data if d[0] == "#"]
                )
                if m != "#"
            ]
        )

        data_df = pd.DataFrame(
            [d.replace("\n", "").split("\t") for d in modified_data if d[0] != "#"]
        )
        for dt in self.df_types:
            # select data type
            df_s = data_df[data_df[1] == f"TYPE_{dt.upper()}"]
            if len(df_s) == 0:
                setattr(self, dt, pd.DataFrame(columns=self.df_type_cols[dt]))
            else:
                # remove empty cols
                na_info = df_s.isna().sum(axis=0) == len(df_s)
                df_s = df_s[
                    [i for i in na_info[na_info == 0].index if i != 1]
                ].reset_index(drop=True)

                if len(df_s.columns) != len(self.df_type_cols[dt]):
                    df_s.columns = self.df_type_cols[dt][: len(df_s.columns)]
                else:
                    df_s.columns = self.df_type_cols[dt]

                # set dtype
                for c in df_s.columns:
                    df_s[c] = df_s[c].astype(self.dtype_dict[dt][c])

                # set DataFrame to attr
                setattr(self, dt, df_s)

    def get_site_info(self, keep_raw=False):
        self.site_info.get_site_info(keep_raw=keep_raw)

    def load_all_data(self, keep_raw=False):
        self.load_df()
        self.get_site_info(keep_raw=keep_raw)

    def __getitem__(self, item):
        if item in self.df_types:
            return getattr(self, item)
        else:
            return None

    def save(self):
        # to be implemented
        with open(f"{self.save_path}/{self.path_id}.pkl", "wb") as f:
            pickle.dump(self, f, -1)


class SiteInfo:
    def __init__(
        self, site_id, floor, input_path="../input/indoor-location-navigation/"
    ):
        self.site_id = site_id
        self.floor = floor
        self.input_path = input_path
        assert Path(input_path).exists(), f"input_path do not exist: {input_path}"

    def get_site_info(self, keep_raw=False):
        floor_info_path = (
            f"{self.input_path}/metadata/{self.site_id}/{self.floor}/floor_info.json"
        )
        with open(floor_info_path, "r") as f:
            self.floor_info = json.loads(f.read())
            self.site_height = self.floor_info["map_info"]["height"]
            self.site_width = self.floor_info["map_info"]["width"]
            if not keep_raw:
                del self.floor_info

        geojson_map_path = (
            f"{self.input_path}/metadata/{self.site_id}/{self.floor}/geojson_map.json"
        )
        with open(geojson_map_path, "r") as f:
            self.geojson_map = json.loads(f.read())
            self.map_type = self.geojson_map["type"]
            self.features = self.geojson_map["features"]

            self.floor_coordinates = self.features[0]["geometry"]["coordinates"]
            self.store_coordinates = [
                self.features[i]["geometry"]["coordinates"]
                for i in range(1, len(self.features))
            ]

            if not keep_raw:
                del self.geojson_map

    def show_site_image(self):
        path = f"{self.input_path}/metadata/{self.site_id}/{self.floor}/floor_image.png"
        plt.imshow(imread(path), extent=[0, self.site_width, 0, self.site_height])

    def draw_polygon(self, size=8, only_floor=False):

        fig = plt.figure()
        ax = plt.subplot(111)

        xmax, xmin, ymax, ymin = self._draw(
            self.floor_coordinates, ax, calc_minmax=True
        )
        if not only_floor:
            self._draw(self.store_coordinates, ax, fill=True)
        plt.legend([])

        xrange = xmax - xmin
        yrange = ymax - ymin
        ratio = yrange / xrange

        self.x_size = size
        self.y_size = size * ratio

        fig.set_figwidth(size)
        fig.set_figheight(size * ratio)
        # plt.show()
        return ax

    def _draw(self, coordinates, ax, fill=False, calc_minmax=False):
        xmax, ymax = -np.inf, -np.inf
        xmin, ymin = np.inf, np.inf
        for i in range(len(coordinates)):
            ndim = np.ndim(coordinates[i])
            if ndim == 2:
                corrd_df = pd.DataFrame(coordinates[i])
                if fill:
                    ax.fill(corrd_df[0], corrd_df[1], alpha=0.7)
                else:
                    corrd_df.plot.line(x=0, y=1, style="-", ax=ax)

                if calc_minmax:
                    xmax = max(xmax, corrd_df[0].max())
                    xmin = min(xmin, corrd_df[0].min())

                    ymax = max(ymax, corrd_df[1].max())
                    ymin = min(ymin, corrd_df[1].min())
            elif ndim == 3:
                for j in range(len(coordinates[i])):
                    corrd_df = pd.DataFrame(coordinates[i][j])
                    if fill:
                        ax.fill(corrd_df[0], corrd_df[1], alpha=0.6)
                    else:
                        corrd_df.plot.line(x=0, y=1, style="-", ax=ax)

                    if calc_minmax:
                        xmax = max(xmax, corrd_df[0].max())
                        xmin = min(xmin, corrd_df[0].min())

                        ymax = max(ymax, corrd_df[1].max())
                        ymin = min(ymin, corrd_df[1].min())
            else:
                assert False, f"ndim of coordinates should be 2 or 3: {ndim}"
        if calc_minmax:
            return xmax, xmin, ymax, ymin
        else:
            return None
