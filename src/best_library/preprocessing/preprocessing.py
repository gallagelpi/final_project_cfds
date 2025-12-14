# ==========================================================
# CLEAN + RECREATE SPLIT FOLDERS
# ==========================================================
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

os.makedirs(f"{WORK_DIR}/train/alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/train/not_alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/val/alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/val/not_alpaca", exist_ok=True)

# Map original folder names to cleaned class names
CLASS_MAP = {
    "alpaca": "alpaca",
    "not alpaca": "not_alpaca"
}
