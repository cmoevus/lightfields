from light_fields.csv_db import CSVDatabase
from light_fields.led import LED
from light_fields.optics import Optics


def test_loading_pairs():
    led_db = CSVDatabase('datasets/LEDs.csv')
    optics_db = CSVDatabase('datasets/Optics.csv')

    pairs = list()
    for l in led_db:
        for o in optics_db.find(led_brand=l['brand'], led_model=l['model']).iterrows():
            led = LED(l, z=0.1)
            optics = Optics(o[1])
            optics.estimate()
            led.optics = optics
            pairs.append(led)

    assert pairs
