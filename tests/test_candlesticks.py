"""Textbook candlestick constructions — each must fire with the right sign."""
import numpy as np
import ta_patterns as tap
from tests.synthetic import fired


def arr(*x):
    return [np.array(z, float) for z in x]


def test_marubozu_white():
    o, h, l, c = arr([10], [11], [10], [11])
    assert tap.marubozu_white(o, h, l, c)[0] == 1


def test_kicking_bullish():
    o, h, l, c = arr([11, 12], [11, 13], [10, 12], [10, 13])
    assert tap.kicking_bullish(o, h, l, c)[1] == 1


def test_kicking_bearish():
    o, h, l, c = arr([10, 9], [11, 9], [10, 8], [11, 8])
    assert tap.kicking_bearish(o, h, l, c)[1] == -1


def test_dark_cloud_cover():
    o, h, l, c = arr([10.0, 13.2], [13.1, 13.4], [9.9, 11.3], [13.0, 11.4])
    assert tap.dark_cloud_cover(o, h, l, c)[1] == -1


def test_piercing_pattern():
    o, h, l, c = arr([13.0, 9.8], [13.1, 12.3], [10.9, 9.7], [11.0, 12.2])
    assert tap.piercing_pattern(o, h, l, c)[1] == 1


def test_engulfing_bearish():
    o, h, l, c = arr([10.0, 12.5], [11.6, 12.6], [9.9, 9.7], [11.5, 9.8])
    assert tap.engulfing_bearish(o, h, l, c)[1] == -1


def test_morning_star():
    o, h, l, c = arr([20.0, 14.6, 14.0], [20.1, 14.8, 18.1],
                     [14.9, 13.9, 13.9], [15.0, 14.2, 18.0])
    assert tap.morning_star(o, h, l, c)[2] == 1


def test_three_white_soldiers():
    o, h, l, c = arr([10, 11, 12.0], [11.6, 12.6, 13.6],
                     [9.9, 10.9, 11.9], [11.5, 12.5, 13.5])
    assert tap.three_white_soldiers(o, h, l, c)[2] == 1


def test_three_line_strike_bullish():
    o, h, l, c = arr([10, 10.5, 11.5, 13.5], [11.1, 12.1, 13.1, 13.6],
                     [9.9, 10.4, 11.4, 9.4], [11, 12, 13, 9.5])
    assert tap.three_line_strike_bullish(o, h, l, c)[3] == 1


def test_three_line_strike_bearish():
    o, h, l, c = arr([13, 12.5, 11.5, 8.8], [13.1, 12.6, 11.6, 13.6],
                     [10.9, 9.9, 8.9, 8.7], [11, 10, 9, 13.5])
    assert tap.three_line_strike_bearish(o, h, l, c)[3] == -1


def test_rising_three_methods():
    o, h, l, c = arr([10, 14, 13.6, 13.2, 13.5], [15.2, 14.1, 13.7, 13.3, 16.1],
                     [9.9, 13.4, 13.0, 12.7, 13.4], [15, 13.5, 13.1, 12.8, 16])
    assert tap.rising_three_methods(o, h, l, c)[4] == 1


def test_falling_three_methods():
    o, h, l, c = arr([15, 11, 11.4, 11.8, 11.5], [15.1, 11.6, 12.0, 12.3, 11.6],
                     [9.8, 10.9, 11.3, 11.7, 9.9], [10, 11.5, 11.9, 12.2, 9.5])
    assert tap.falling_three_methods(o, h, l, c)[4] == -1


def test_three_stars_south():
    o, h, l, c = arr([20, 17, 15.3], [20.1, 17.05, 15.32],
                     [16.9, 15.4, 13.98], [17, 15.5, 14.0])
    assert tap.three_stars_south(o, h, l, c)[2] == 1


def test_concealing_baby_swallow():
    o, h, l, c = arr([20, 19, 17.5, 18.5], [20, 19, 18.2, 18.6],
                     [19, 18, 16.9, 16.4], [19, 18, 17, 16.5])
    assert tap.concealing_baby_swallow(o, h, l, c)[3] == 1
