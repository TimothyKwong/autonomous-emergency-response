import os
import csv

import logging

class MsgLogger:

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    def __init__(self):
        self._messages = []

    def log(self, msg: str):
        logging.info(msg)

    def collect(self, msg: str):
        """
        Collect a message to be logged later.
        Accepts an f-string or normal string.
        """
        self._messages.append(msg)

    def logAll(self):
        """
        Logs all collected messages using logging.info and clears the collector.
        """
        if not self._messages:
            return
        combined_msg = "\n".join(self._messages)
        logging.info("\n" + combined_msg)
        self._messages.clear()

class DataLogger:
    def __init__(self, measurements, filename: str='light'):
        self.measurements = measurements
        self.filename = filename
        self.out_dir = "./data/raw"
        self.init_csv()

    def init_csv(self):
        # Ensure filename extension has csv
        filename = self.filename
        
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        self.csv_path = os.path.join(self.out_dir, filename)

        # Extract measurement attribute names as headers
        headers = [attr_name for attr_name, _ in self.measurements.__dict__.items()]

        # Save headers to CSV (write, overwrite mode)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def save_csv(self):
        if not hasattr(self, "csv_path"):
            raise ValueError("CSV file not initialized. Call init_csv() first.")

        # Extract latest measurement values
        columns = []
        for _, attr_value in self.measurements.__dict__.items():
            latest_value = attr_value[-1]
            columns.append(latest_value)

        # Write as a new row (append mode)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
