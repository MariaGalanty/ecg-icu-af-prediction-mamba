import torch
import os

# Paths
AF_DIR = "/mnt/maria_amc_storage/qurai/ProjectData/ICU lab/AF prediction/snowflake/AF_petection_project_lead_II_v1.5/Pre_AF_SR/"
SR_DIR = "/mnt/maria_amc_storage/qurai/ProjectData/ICU lab/AF prediction/snowflake/AF_petection_project_lead_II_v1.5/Pure_SR_Controls/"

# Hyperparameters
D_MODEL = 64
N_LAYERS = 2
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 0.08
EPOCHS = 3
PATIENCE = 10
WINDOW_MINUTES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
