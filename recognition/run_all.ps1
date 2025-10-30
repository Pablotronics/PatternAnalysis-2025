param(
  [string]$DataRoot = "C:\Users\User\Desktop\Final project comp3710\keras_slices_data",
  [int]$Epochs = 2,
  [int]$BatchSize = 4,
  [switch]$SavePNGs,
  [int]$NumSamples = 9
)

Write-Host "[INFO] Using data root: $DataRoot"
Write-Host "[INFO] Training (Project 3: U-Net + CAN)..."
python Project2\train.py --data_root "$DataRoot" --epochs $Epochs --batch_size $BatchSize

Write-Host "[INFO] Predicting / evaluating..."
$predictArgs = @(
  "Project2\predict.py",
  "--data_root", "$DataRoot",
  "--num_samples", $NumSamples
)
if ($SavePNGs) { $predictArgs += "--save_pngs" }

python @predictArgs
