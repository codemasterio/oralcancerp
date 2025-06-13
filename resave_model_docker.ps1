# Create output directory if it doesn't exist
$outputDir = "./model/checkpoints"
if (-not (Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t oral-cancer-resave -f Dockerfile.resave .

# Run the container and copy the output files
Write-Host "Running container to re-save model..."
docker run --rm \
    -v "${PWD}/model/checkpoints:/output" \
    oral-cancer-resave \
    sh -c "python resave.py && cp model_np1243.pkl scaler_np1243.pkl /output/"

Write-Host "Done! Check the model/checkpoints directory for the re-saved files."
