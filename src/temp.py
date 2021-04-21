import numpy as np
from utils.common import load_pickle


def main():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    bssid_map = load_pickle("../data/label_encode/map_bssid.pkl", verbose=False)

    for path_id, gdf in waypoint.groupby("path"):
        path_id = "5ce215be915519000851776a"
        print(path_id)
        print(gdf)

        ts_waypoint = gdf["timestamp"].to_numpy()
        min_idx = gdf.index.min()
        max_idx = gdf.index.max()

        print(min_idx, max_idx)

        feature = load_pickle(f"../data/working/{path_id}.pkl")
        wifi = feature.wifi.copy()
        # print(wifi.head())

        for i, row in gdf.iterrows():
            n_diff = 1
            # ts_current_wp = ts_waypoint[i]
            ts_pre_wp = ts_waypoint[i - n_diff] if (i - n_diff) >= min_idx else None
            ts_post_wp = ts_waypoint[i + n_diff] if (i + n_diff) <= max_idx else None

            ts_wifi = wifi["timestamp"].to_numpy()
            pre_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_pre_wp is None
                else ts_wifi > ts_pre_wp
            )
            post_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_post_wp is None
                else ts_wifi < ts_post_wp
            )

            _wifi = wifi[pre_flag & post_flag].copy()
            _wifi.sort_values(by="rssi", ascending=False, inplace=True)
            _wifi.drop_duplicates(subset=["bssid"], keep="first", inplace=True)

            print(_wifi.head())

        break


if __name__ == "__main__":
    main()
