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
