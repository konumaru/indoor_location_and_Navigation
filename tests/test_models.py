import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.simplefilter("ignore")

from src import models


def test_loss_function():
    batch_size = 64
    pos = torch.rand(size=(batch_size, 2), requires_grad=True)
    pos_hat = pos.clone()

    floor = torch.randint(14, size=(batch_size,))
    floor_hat = torch.rand(size=(batch_size, 14), requires_grad=True)

    loss_fn = models.MeanAbsolutePositionLoss()
    loss = loss_fn(pos_hat, pos, floor_hat, floor)
    loss.backward()


def test_evaluation_function():
    batch_size = 64
    pos = torch.rand(size=(batch_size, 2), requires_grad=True)
    pos_hat = pos.clone()

    floor = torch.randint(14, size=(batch_size,))
    floor_hat = floor.clone()

    eval_fn = models.MeanPositionLoss()
    metric = eval_fn(pos_hat, pos, floor_hat, floor)

    assert metric == 0


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


def get_wifi_feature(batch_size: int = 32, seq_len: int = 20):
    site = torch.randint(205, size=(batch_size,))
    floor = torch.randint(14, size=(batch_size,))

    bssid = torch.randint(238860, size=(batch_size, seq_len))
    rssi = torch.rand(size=(batch_size, seq_len))
    freq = torch.rand(size=(batch_size, seq_len))
    last_seen_ts = torch.rand(size=(batch_size, seq_len))
    return (site, floor, bssid, rssi, freq, last_seen_ts)


def test_wifi_model():
    seq_len = 20
    batch_size = 32

    input_wifi = get_wifi_feature(batch_size, seq_len)

    model = models.WifiModel(seq_len=seq_len)
    z = model(input_wifi)

    assert z.size(0) == batch_size


def get_beacon_feature(batch_size: int = 100, seq_len: int = 20):
    uuid = torch.randint(100, size=(batch_size, seq_len))
    tx_power = torch.rand(size=(batch_size, seq_len))
    rssi = torch.rand(size=(batch_size, seq_len))
    return (uuid, tx_power, rssi)


def test_beacon_model():
    batch_size = 32
    input_beacon = get_beacon_feature(batch_size)
    model = models.BeaconModel()
    z = model(input_beacon)

    assert z.size(0) == batch_size


def get_acce_feature(batch_size: int = 100, seq_len: int = 20):
    past_x = torch.rand(size=(batch_size, seq_len))
    past_y = torch.rand(size=(batch_size, seq_len))
    past_z = torch.rand(size=(batch_size, seq_len))

    feat_x = torch.rand(size=(batch_size, seq_len))
    feat_y = torch.rand(size=(batch_size, seq_len))
    feat_z = torch.rand(size=(batch_size, seq_len))
    return (past_x, past_y, past_z, feat_x, feat_y, feat_z)


def test_acce_model():
    batch_size = 32
    input_acce = get_acce_feature(batch_size, seq_len=100)
    model = models.AccelemoterModel()
    z = model(input_acce)

    print(z.shape)

    assert z.size(0) == batch_size


def test_indoor_model():
    batch_size = 32
    input_build = get_build_feature(batch_size)
    input_wifi = get_wifi_feature(batch_size, seq_len=20)
    input_beacon = get_beacon_feature(batch_size)
    input_acce = get_acce_feature(batch_size, seq_len=100)

    floor = torch.randint(14, size=(batch_size,))
    waypoint = torch.rand(size=(batch_size, 2))

    x = (input_build, input_wifi, input_beacon, input_acce)
    y = (floor, waypoint)

    model = models.InddorModel(wifi_seq_len=20)
    floor_hat, pos_hat = model(x)

    loss_fn = models.MeanAbsolutePositionLoss()
    loss = loss_fn(pos_hat, y[1], floor_hat, y[0])
    loss.backward()

    eval_fn = models.MeanPositionLoss()
    metric = eval_fn(pos_hat, y[1], torch.argmax(floor_hat, dim=1), y[0])
