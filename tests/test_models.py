import torch
import warnings

warnings.simplefilter("ignore")

from src import models


def get_build_feature(batch_size: int = 100):
    site_id = torch.randint(100, size=(batch_size, 1))
    return site_id


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
    return (bssid, rssi, freq)


def test_wifi_model():
    seq_len = 100
    batch_size = 100
    input_wifi = get_wifi_feature(batch_size, seq_len)

    model = models.WifiModel()
    z = model(input_wifi)

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
    z = model(x)

    print(z)

    loss_fn = models.MeanPositionLoss()
    loss = loss_fn(z, y)
    # loss_fn = models.RMSELoss()
    # loss = loss_fn(z, y[1])
    # loss.backward()
