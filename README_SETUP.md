## Backend Setup & Run Instructions

Follow these steps to run the backend (Flask + YOLOv8 traffic counter). Ensure Python 3.10+.

1. Create virtual environment (only first time):
	python3 -m venv venv

2. Activate virtual environment:
	macOS / Linux:
	  source venv/bin/activate
	Windows (PowerShell):
	  venv\\Scripts\\Activate.ps1

3. Install backend dependencies (file name may be `requirements_backend.txt` or `requirement_backend.txt` if you created the alternate one):
	pip install -r requirements_backend.txt || pip install -r requirement_backend.txt

4. (Optional) Ensure YOLO weight files exist in YOLOv8-Traffic-Counter-main/ (e.g. yolov8l.pt, yolov8n.pt). Place them there if missing.

5. Run the backend server:
	python simple_backend.py

6. Server will start on:
	http://localhost:5003

7. Frontend (React) should point API calls to port 5003 (adjust fetch base URL if needed). The app currently expects `http://localhost:5003`.

8. To stop:
	Press CTRL+C

9. To deactivate virtual environment after finishing:
	deactivate

### Quick One-Liner (after first setup)
source venv/bin/activate && python simple_backend.py

---
## Frontend Setup & Run Instructions

1. Change into frontend folder:
	cd frontend
2. Install dependencies (first time):
	npm install
3. Start development server (default http://localhost:3000):
	npm start
4. Ensure CORS origin (localhost:3000) is allowed (already configured in backend CORS list).
5. If you change backend port, update any fetch URLs accordingly.

### Environment / Config Notes
- YOLO weights expected in: `YOLOv8-Traffic-Counter-main/`
- Default model key: `yolov8l` (configured in backend global `current_model_name`).
- API endpoints: see `/api/debug/routes` (added for troubleshooting) or code in `simple_backend.py`.

### Optional Cleanup
Legacy `sort.py` and `YOLOv8-Traffic-Counter-main/Car Counter.py` are unused by `simple_backend.py`. You can archive or delete them to slim dependencies.

