# Task 3.5 - the full run plan. Each run differs from another in ONE meaningful change.
# Runs one after another; safe to leave unattended (e.g. overnight).
#   .\scripts\run_experiments.ps1
# Every run logs to DagsHub (needs .env) and saves figures in reports\runs\<run-name>\.

$runs = @(
  # name                 arguments                                              # what changes
  @("cnn_lr1e-3",        "--model cnn --lr 1e-3 --register"),                   # baseline CNN
  @("cnn_lr3e-4",        "--model cnn --lr 3e-4"),                              # learning rate
  @("cnn_noaug",         "--model cnn --lr 1e-3 --no-augment"),                 # augmentation off
  @("cnn_autolog",       "--model cnn --lr 1e-3 --log-mode auto"),              # autologging (Task 3.3)
  @("mnv2_160",          "--model transfer --register"),                        # backbone: MobileNetV2
  @("mnv2_160_frozen",   "--model transfer --no-finetune"),                     # no unfreezing
  @("mnv2_128",          "--model transfer --img-size 128"),                    # image size
  @("effb0_160",         "--model transfer --backbone efficientnetb0 --register")  # backbone: EfficientNetB0
)

foreach ($r in $runs) {
  Write-Host "`n==================== $($r[0]) ====================" -ForegroundColor Cyan
  $cmd = "python scripts\train.py --run-name $($r[0]) $($r[1])"
  Write-Host $cmd
  Invoke-Expression $cmd
  if ($LASTEXITCODE -ne 0) { Write-Host "RUN $($r[0]) FAILED (exit $LASTEXITCODE)" -ForegroundColor Red }
}
python scripts\compare_runs.py
