# Trail Running Time Predictor

Machine learning application that predicts trail running completion times based on route characteristics and historical Strava data.

## 🚀 Features

- **GPX Route Analysis**: Upload GPX files to analyze elevation, distance, and terrain
- **ML Predictions**: Predict completion times based on trained models
- **Automated Data Pipeline**: Monthly updates from Strava API via Azure Functions
- **Automatic Data Initialization**: Self-sufficient setup on first run

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  - GPX Upload UI                                                │
│  - Prediction Display                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + Python)                    │
│  - Feature Extraction (GPX → Features)                          │
│  - ML Model Training & Inference                                │
│  - Automatic Data Initialization on Startup                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Pipeline                               │
│  ┌──────────────────┐        ┌──────────────────┐              │
│  │ Azure Function   │───────>│ Azure Blob       │              │
│  │ (Monthly Updates)│        │ Storage          │              │
│  └──────────────────┘        └──────────────────┘              │
│           ▲                                                     │
│           │                                                     │
│  ┌────────┴─────────┐                                          │
│  │ Strava API       │                                          │
│  │ (Activities)     │                                          │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
trail-running-time-predictor/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── data_ingestion/    # Strava API client & pipeline
│   │   ├── feature_engineering/ # GPX feature extraction
│   │   ├── utils/             # Blob storage, data initialization
│   │   ├── config/            # Settings & configuration
│   │   └── main.py            # FastAPI app entry point
│   ├── scripts/               # Utility scripts
│   └── requirements.txt
├── azure_functions/           # Azure Functions for monthly updates
│   └── strava_monthly_update/
├── frontend/                  # React frontend (TBD)
├── SECURITY_GUIDE.md         # 🔐 Security best practices
├── SETUP_AZURE_MONTHLY_UPDATE.md  # Azure setup guide
├── AUTOMATIC_DATA_INITIALIZATION.md  # Data initialization docs
└── check_secrets.py          # Pre-commit security check
```

## 🔐 Security First

**IMPORTANT**: This project handles sensitive credentials (API keys, connection strings).

### Before Your First Commit:

1. **Verify `.gitignore` is working**:
   ```bash
   git status
   ```
   Should **NOT** show:
   - `local.settings.json`
   - `.env` files

2. **Run security check**:
   ```bash
   python check_secrets.py
   ```

3. **Read the security guide**:
   - See [SECURITY_GUIDE.md](SECURITY_GUIDE.md) for complete instructions

### Protected Files (Already in `.gitignore`):
- ✅ `azure_functions/local.settings.json` - Azure Functions credentials
- ✅ `.env` and `.env.local` - Backend environment variables
- ✅ `*.log` - Log files
- ✅ `backend/data/*.parquet` - Training data

## 🚀 Quick Start

**📖 TL;DR?** Vezi [QUICK_START.md](QUICK_START.md) pentru ghid rapid în 5 minute!

### Prerequisites

- Python 3.9+
- Node.js 18+ (for Azure Functions Core Tools - optional)
- Azure Account (for cloud storage and functions)
- Strava API credentials (deja configurate în `.env`)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (see SECURITY_GUIDE.md)
# Windows PowerShell:
$env:AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
$env:STRAVA_CLIENT_ID="your_client_id"
$env:STRAVA_CLIENT_SECRET="your_client_secret"
$env:STRAVA_REFRESH_TOKEN="your_refresh_token"

# Start the backend
uvicorn app.main:app --reload
```

On first startup, the backend will automatically:
- Check Azure Blob Storage for training data
- If no data exists, fetch ALL Strava activities and create initial dataset
- Upload dataset to Azure Blob Storage

See [AUTOMATIC_DATA_INITIALIZATION.md](AUTOMATIC_DATA_INITIALIZATION.md) for details.

### 2. Azure Functions Setup (Optional - for monthly updates)

See [SETUP_AZURE_MONTHLY_UPDATE.md](SETUP_AZURE_MONTHLY_UPDATE.md) for complete setup guide.

Quick version:
```bash
cd azure_functions

# Copy template
copy local.settings.json.example local.settings.json

# Edit local.settings.json with your credentials
# (This file is git-ignored)

# Test locally (optional)
func start

# Deploy to Azure
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

## 📚 Documentation

- [SECURITY_GUIDE.md](SECURITY_GUIDE.md) - **Start here!** Credential management & best practices
- [AUTOMATIC_DATA_INITIALIZATION.md](AUTOMATIC_DATA_INITIALIZATION.md) - How automatic data setup works
- [SETUP_AZURE_MONTHLY_UPDATE.md](SETUP_AZURE_MONTHLY_UPDATE.md) - Complete Azure Functions setup
- [azure_functions/README.md](azure_functions/README.md) - Azure Functions technical details

## 🛠️ Development Workflow

### Before Committing:

```bash
# 1. Check what files will be committed
git status

# 2. Run security check
python check_secrets.py

# 3. If all clear, commit
git add .
git commit -m "Your commit message"
git push
```

### Testing Locally:

```bash
# Test backend
cd backend
uvicorn app.main:app --reload

# Test data initialization
cd backend
python scripts/test_data_initialization.py

# Test Azure Function logic (without deploying)
cd backend
python scripts/test_monthly_update_logic.py
```

## 🔧 Configuration

### Environment Variables

**Backend** (set these before starting):
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection
- `STRAVA_CLIENT_ID` - Strava API client ID
- `STRAVA_CLIENT_SECRET` - Strava API client secret
- `STRAVA_REFRESH_TOKEN` - Strava API refresh token

**Azure Functions** (set in Azure Portal → Function App → Configuration):
- Same variables as above

See [SECURITY_GUIDE.md](SECURITY_GUIDE.md) for how to set these securely.

## 📊 Data Pipeline

### Automatic Initialization (First Run)
1. Backend starts up
2. Checks Azure Blob Storage for training data
3. If no data exists:
   - Fetches ALL Strava activities (Run type, elevation >= 100m)
   - Extracts features from each activity
   - Uploads initial dataset to Blob Storage

### Monthly Updates (Azure Function)
1. Timer triggers on 1st of month at 00:00 UTC
2. Downloads existing dataset from Blob Storage
3. Fetches NEW activities from Strava (since last update)
4. Appends new data and uploads back to Blob Storage

## 🧪 Testing

```bash
# Run security check
python check_secrets.py

# Test data initialization
cd backend
python scripts/test_data_initialization.py

# Test monthly update logic
cd backend
python scripts/test_monthly_update_logic.py

# Start backend
cd backend
uvicorn app.main:app --reload
```

## 🐛 Troubleshooting

### "No module named 'app'"
```bash
# Make sure you're in the backend directory
cd backend
# And virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### "Azure Storage connection string is required"
Set the environment variable:
```powershell
# Windows PowerShell
$env:AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
```

### "local.settings.json appears in git status"
This should NOT happen (file is git-ignored). If it does:
```bash
# Remove from staging
git reset HEAD azure_functions/local.settings.json

# If you already committed it, see SECURITY_GUIDE.md for recovery steps
```

## 💰 Cost Estimation

- **Azure Functions**: FREE (within 1M executions/month)
- **Azure Blob Storage**: ~$0.01/month (1-10 MB storage)
- **Total**: ~$0.01/month

## 🤝 Contributing

Before contributing:

1. ✅ Read [SECURITY_GUIDE.md](SECURITY_GUIDE.md)
2. ✅ Run `python check_secrets.py` before commits
3. ✅ Never commit credentials or API keys
4. ✅ Use `local.settings.json.example` as template (not actual `local.settings.json`)

## 📝 License

[Add your license here]

## 🔗 Useful Links

- [Strava API Documentation](https://developers.strava.com/)
- [Azure Functions Documentation](https://learn.microsoft.com/en-us/azure/azure-functions/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Remember**: Security first! Always run `python check_secrets.py` before pushing to GitHub. 🔐
