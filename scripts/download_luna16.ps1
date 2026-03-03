param(
    [string]$TargetDir = "data/LUNA16",
    [switch]$Extract
)

$ErrorActionPreference = "Stop"

$targetPath = Resolve-Path -Path . | ForEach-Object { Join-Path $_ $TargetDir }
New-Item -ItemType Directory -Force -Path $targetPath | Out-Null

$files = @(
    @{ Name = "annotations.csv"; Url = "https://zenodo.org/api/records/3723295/files/annotations.csv/content" },
    @{ Name = "subset0.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset0.zip/content" },
    @{ Name = "subset1.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset1.zip/content" },
    @{ Name = "subset2.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset2.zip/content" },
    @{ Name = "subset3.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset3.zip/content" },
    @{ Name = "subset4.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset4.zip/content" },
    @{ Name = "subset5.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset5.zip/content" },
    @{ Name = "subset6.zip"; Url = "https://zenodo.org/api/records/3723295/files/subset6.zip/content" },
    @{ Name = "subset7.zip"; Url = "https://zenodo.org/api/records/4121926/files/subset7.zip/content" },
    @{ Name = "subset8.zip"; Url = "https://zenodo.org/api/records/4121926/files/subset8.zip/content" },
    @{ Name = "subset9.zip"; Url = "https://zenodo.org/api/records/4121926/files/subset9.zip/content" }
)

Write-Host "Target: $targetPath"

foreach ($file in $files) {
    $destination = Join-Path $targetPath $file.Name
    if (Test-Path $destination) {
        Write-Host "[SKIP] $($file.Name) already exists"
    }
    else {
        Write-Host "[DOWN] $($file.Name)"
        Start-BitsTransfer -Source $file.Url -Destination $destination -DisplayName "LUNA16-$($file.Name)"
        Write-Host "[DONE] $($file.Name)"
    }

    if ($Extract -and $file.Name -like "subset*.zip") {
        $extractFolder = Join-Path $targetPath ($file.Name -replace "\.zip$", "")
        if (-not (Test-Path $extractFolder)) {
            Write-Host "[UNZIP] $($file.Name) -> $extractFolder"
            Expand-Archive -Path $destination -DestinationPath $extractFolder -Force
            Write-Host "[OK] Extracted $($file.Name)"
        }
        else {
            Write-Host "[SKIP] Extract folder exists: $extractFolder"
        }
    }
}

Write-Host "LUNA16 download completed."
