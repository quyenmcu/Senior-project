# IrisMind

**IrisMind: An Intelligent Decision-Support System for Fatigue-Aware
Usage-Based Insurance Premium Prediction**

IrisMind is a Django web application that combines browser-camera fatigue
detection, smartphone telematics, driver risk summaries, and transparent
insurance-premium estimates. The current premium forecast is a formula-based
research baseline—not an insurer quote or a trained MLP/LSTM prediction.

## Features

- Registration, login, logout, and email password reset
- Private driver dashboard and staff-only administration overview
- Browser-camera fatigue monitor using MediaPipe Face Mesh
- Active, Drowsy, Sleeping, and Yawning event histories
- Protected fatigue snapshots
- Smartphone telemetry ingestion and driving-behavior summaries
- Daily fatigue graphs and weekly/monthly premium projections
- Historical/projected premium time-series graph
- Optional, clearly labeled Kaggle demonstration panels
- Authentication, privacy, ingestion, risk, and dashboard tests

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or 3.12
- Webcam and a modern browser with camera permission
- Gmail App Password only when real password-reset email is required

No GPU is required. MediaPipe and OpenCV run on the CPU.

## Windows PowerShell setup

```powershell
git clone https://github.com/quyenmcu/Senior-project.git
cd Senior-project
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,cv]"
Copy-Item .env.example .env
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

Open:

- Home: `http://127.0.0.1:8000/`
- Registration: `http://127.0.0.1:8000/accounts/register/`
- Dashboard: `http://127.0.0.1:8000/dashboard/`
- Fatigue monitor: `http://127.0.0.1:8000/fatigue/`
- Staff overview: `http://127.0.0.1:8000/staff/overview/`

Keep PowerShell open while using the application.

## macOS/Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cv]"
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## Fatigue monitor

Allow camera permission when prompted. The first 50 valid face frames calibrate
the eye-aspect-ratio baseline for that driving session. Snapshots are stored
only when Drowsy, Sleeping, or Yawning transitions occur.

## Password-reset email

The local default prints reset messages and links in the terminal. This permits
testing without an email account. For Gmail delivery, edit `.env`, configure
`EMAIL_HOST_USER`, and set `EMAIL_HOST_PASSWORD` to a Google App Password. Never
use or commit the normal Gmail account password.

## Optional demo telemetry

Place licensed copies of `train_motion_data.csv` and `features_14.csv` inside
`demo_data/`. They are presented only as demonstration data, are not inserted
into personal records, and never affect a user's risk or premium. The app works
without them.

Real telemetry must arrive through the authenticated API with the correct user,
driving session, and recorded timestamp. Never associate separate datasets by
row position.

## Premium calculation

The current research baseline uses:

```text
Total risk = (fatigue risk + driving risk) / 2
Annual premium = base premium + total risk × risk rate × annual distance
```

Weekly and monthly views show the estimated annual premium at future dates—not
weekly or monthly payment amounts. With insufficient history, the forecast
stays flat rather than inventing a trend.
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
pytest -q
python manage.py check
```

## Security and privacy

- Driver queries are scoped through the authenticated user.
- Staff pages use Django staff authorization.
- Snapshots are served through ownership checks, not public media URLs.
- `.env`, `db.sqlite3`, face media, model files, archives, and local datasets
  are excluded from Git.
- Do not commit Gmail credentials or a production Django secret key.

## Project structure

```text
apps/accounts/       Registration and account flows
apps/driving/        Sessions, fatigue/telemetry models, services and APIs
apps/dashboard/      User/staff dashboards and demo-data summaries
config/              Django configuration
templates/           Django HTML templates
static/              CSS, JavaScript and public images
tests/               Automated test suite
docs/                Architecture documentation
```

## Next stages

- Connect a smartphone client to the authenticated telemetry API.
- Collect longitudinal, consented driver data.
- Train and validate an MLP risk model when enough labeled records exist.
- Treat LSTM forecasting as experimental until each driver has sufficient
  consecutive weekly or monthly records.
pytest
```

## Next stage

Connect a smartphone sensor client to the authenticated telemetry ingestion endpoint. Once real
samples exist for a user, the dashboard automatically stops showing the Kaggle demo panels.
