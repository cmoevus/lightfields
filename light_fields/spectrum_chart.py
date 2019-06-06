import matplotlib.pyplot as plt

from csv_db import CSVDatabase


def generate_spectrum_chart_from_csv(csv_file):
    led_db = CSVDatabase('Light/LEDs.csv')
    led_db.db.set_index(['brand', 'model', 'serial_number'])

    pairs = list()
    led_record = led_db.find(serial_number='B07D9HWHY6')[0]
    led = LED(led_record, z=0.1)

    plt.figure()
    x = np.linspace(400, 700, 1000)
    plt.plot(x, sp_filters.gaussian_filter1d(pair.spectrum(x), 5))

    plt.savefig('out.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    args = parser.parse_args()

    generate_spectrum_chart_from_csv(args.csv_file)
