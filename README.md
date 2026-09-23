# IrisMind

Fatigue-aware usage-based insurance platform built as a Django monolith.

## Included

- New-user registration and existing-user login
- Private, per-user driver dashboard
- Fatigue events and smartphone telemetry models
- Driving-behavior aggregation
- Current premium calculation
- Six-month baseline premium projection
- Weekly and monthly premium forecasts with a historical/projected time-series graph
- Four-series daily fatigue chart for drivers and staff
- Clearly labeled Kaggle telemetry demonstration panels
- Ownership-isolation tests
- Responsive system-architecture diagram and two website demo visuals

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -e ".[dev,cv]"
cp .env.example .env
python manage.py makemigrations driving
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/accounts/register/`.

The embedded fatigue monitor is at `http://127.0.0.1:8000/fatigue/`. Allow camera
permission when prompted. The first 50 valid face frames calibrate neutral EAR and head
pose for that session.

After neutral calibration, tilt left with both eyes open and click **Calibrate Left**.
Hold the pose for 30 frames, then repeat on the right with **Calibrate Right**. IrisMind
uses pitch for Forward/Backward, roll for Left/Right, and the side-specific open-eye
baselines to detect eye closure accurately while the head is tilted. Calibration values
exist only in memory for the current authenticated driving session.

Create a staff account for the administration overview:

```bash
python manage.py createsuperuser
```

Then log in and open `http://127.0.0.1:8000/staff/overview/`.

## Password-reset email

The local default prints reset emails and links in the PowerShell window so the complete flow
can be tested without an email account. To send real email through Gmail, copy `.env.example`
to `.env`, set `EMAIL_HOST_USER`, and use a Google App Password for
`EMAIL_HOST_PASSWORD`. Do not use or commit your normal Gmail password.

Fatigue snapshots are saved only for Drowsy, Sleeping and Yawning status transitions. They are
served through an authenticated ownership check rather than a public media URL.

## Demo telemetry data

`demo_data/train_motion_data.csv` and `demo_data/features_14.csv` are public demonstration
datasets. They are shown only when the signed-in user has no real telemetry. Demo rows are not
inserted into the database and never affect the user's risk score or premium. The raw dataset's
timestamp is a sample counter, so IrisMind does not present it as a calendar date.

Real user telemetry must continue to arrive through the authenticated ingestion API with the
correct user, driving session, and recorded timestamp.

## Verification

```bash
ruff check .
pytest
```

## Next stage

Connect a smartphone sensor client to the authenticated telemetry ingestion endpoint. Once real
samples exist for a user, the dashboard automatically stops showing the Kaggle demo panels.
