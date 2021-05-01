import torch
import warnings

warnings.simplefilter("ignore")

from src import models


def get_build_feature(batch_size: int = 100):
    site_id = torch.randint(100, size=(batch_size, 1))
    floor = torch.randint(10, size=(batch_size, 1), dtype=torch.long)
    return (site_id, floor)


def test_build_model():
    batch_size = 100
    input_build = get_build_feature(batch_size)

    model = models.BuildModel()
    z = model(input_build)

    assert z.size(0) == batch_size


def get_wifi_feature(batch_size: int = 100, seq_len: int = 20):
    bssid = torch.randint(100, size=(batch_size, seq_len))
    rssi = torch.rand(size=(batch_size, seq_len))
    freq = torch.rand(size=(batch_size, seq_len))
    last_seen_ts = torch.rand(size=(batch_size, seq_len))
    return (bssid, rssi, freq, last_seen_ts)


def test_wifi_model():
    seq_len = 100
    batch_size = 100

    input_build = get_build_feature(batch_size)
    model = models.BuildModel()
    x_build = model(input_build)

    input_wifi = get_wifi_feature(batch_size, seq_len)

    model = models.WifiModel()
    z = model(input_wifi, x_build)

    assert z.size(0) == batch_size


def get_beacon_feature(batch_size: int = 100, seq_len: int = 20):
    uuid = torch.randint(100, size=(batch_size, seq_len))
    tx_power = torch.rand(size=(batch_size, seq_len))
    rssi = torch.rand(size=(batch_size, seq_len))
    return (uuid, tx_power, rssi)


def test_beacon_model():
    batch_size = 100
    input_beacon = get_beacon_feature(batch_size)
    model = models.BeaconModel()
    z = model(input_beacon)

    assert z.size(0) == batch_size


def test_indoor_model():
    batch_size = 100
    input_build = get_build_feature(batch_size)
    input_wifi = get_wifi_feature(batch_size, seq_len=100)
    input_beacon = get_beacon_feature(batch_size)

    floor = torch.randint(14, size=(batch_size, 1))
    waypoint = torch.rand(size=(batch_size, 2))

    x = (input_build, input_wifi, input_beacon)
    y = (floor, waypoint)

    model = models.InddorModel()
    floor_hat, pos_hat = model(x)

    loss_fn = models.MeanAbsolutePositionLoss()
    loss = loss_fn(pos_hat, y[1], floor_hat, torch.flatten(y[0]))
    loss.backward()

    eval_fn = models.MeanPositionLoss()
    metric = eval_fn(pos_hat, y[1], torch.argmax(floor_hat, dim=1), y[0])
