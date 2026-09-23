# Optional demonstration data

The dashboard can preview two public demonstration datasets when a signed-in
user has no real smartphone telemetry:

- `train_motion_data.csv`: raw accelerometer/gyroscope samples with `Class`
  labels (`NORMAL`, `AGGRESSIVE`, `SLOW`).
- `features_14.csv`: statistical sensor windows with `Target` values 1–4.

The CSV files are intentionally excluded from Git. Add them to this directory
locally only after verifying their Kaggle source and redistribution license.
They are never inserted into a user's database records and never affect risk or
premium calculations.

The website works without these files; the demonstration section is simply
hidden.
