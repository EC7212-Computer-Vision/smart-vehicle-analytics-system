## Backend Setup & Run Instructions

Follow these steps to run the backend (Flask + YOLOv8 traffic counter):

1. Create virtual environment (only first time):
	python3 -m venv venv

2. Activate virtual environment:
	macOS / Linux:
	  source venv/bin/activate
	Windows (PowerShell):
	  venv\\Scripts\\Activate.ps1

3. Install backend dependencies:
	pip install -r requirements_backend.txt

4. (Optional) Ensure YOLO weight files exist in YOLOv8-Traffic-Counter-main/ (e.g. yolov8l.pt, yolov8n.pt). Place them there if missing.

5. Run the backend server:
	python simple_backend.py

6. Server will start on:
	http://localhost:5003

7. Frontend (React) should point API calls to port 5003 (adjust fetch base URL if needed).

8. To stop:
	Press CTRL+C

9. To deactivate virtual environment after finishing:
	deactivate

### Quick One-Liner (after first setup)
source venv/bin/activate && python simple_backend.py

**🚗 Happy Traffic Monitoring! 📊**