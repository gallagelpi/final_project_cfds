# ==========================================================
# LOAD DATA
# ==========================================================
train_dataset = datasets.ImageFolder(os.path.join(WORK_DIR, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(WORK_DIR, "val"),   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

print("Classes detected:", train_dataset.classes)
