Add-Type -AssemblyName System.Web

# Dynamic Paths Configuration relative to script location
$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$scriptPath   = Join-Path $scriptDir "Backend.py"
$logPath      = Join-Path $scriptDir "reconstruct_pipeline.log"
$manifestPath = Join-Path $scriptDir "SCENES\scenes_manifest.json"
$htmlPath     = Join-Path $scriptDir "Frontend.html"
$videosDir    = Join-Path $scriptDir "VIDEOS"
$configPath   = Join-Path $scriptDir "config.json"

if (!(Test-Path $videosDir)) { New-Item -ItemType Directory -Path $videosDir | Out-Null }

# Dynamically locate Python 3.10 runtime
function Get-Python310Path {
    $searchRoots = @($scriptDir, (Get-Location).Path, [System.Environment]::GetFolderPath('UserProfile'))
    $relPaths = @(".venv\Scripts\python.exe", "venv\Scripts\python.exe", "env\Scripts\python.exe")

    foreach ($root in $searchRoots) {
        foreach ($rel in $relPaths) {
            $candidate = Join-Path $root $rel
            if (Test-Path $candidate) {
                $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($ver -eq "3.10") { return $candidate }
            }
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pyPath = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
        if ($pyPath -and (Test-Path $pyPath)) { return $pyPath }
    }

    $pathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPython) {
        $ver = & $pathPython.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -eq "3.10") { return $pathPython.Source }
    }

    return $null
}

$pythonExe = Get-Python310Path

if (-not $pythonExe) {
    Write-Host "CRITICAL: Could not locate Python 3.10 environment!" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python Runtime: $pythonExe" -ForegroundColor Green

$global:pyProcess = $null

$httpListener = New-Object System.Net.HttpListener
$httpListener.Prefixes.Add("http://localhost:8080/")
try {
    $httpListener.Start()
} catch {
    Write-Host "Port 8080 is already in use." -ForegroundColor Red
    exit 1
}

Write-Host "Reconstruction Hub Server active on http://localhost:8080/" -ForegroundColor Cyan

# Recursive process termination: kills all spawned child processes (COLMAP, FFmpeg, Torch)
function Stop-Backend {
    if ($global:pyProcess -and -not $global:pyProcess.HasExited) {
        Write-Host "Terminating Python process tree..." -ForegroundColor Yellow
        try {
            & taskkill /PID $global:pyProcess.Id /T /F | Out-Null
        } catch {
            # Process was already terminating or detached
        }
        $global:pyProcess = $null
    }
}

function Start-Backend ([string]$videoPath, [bool]$resume = $false) {
    Stop-Backend
    [System.Threading.Thread]::Sleep(250)

    $cleanVideoPath = if ($videoPath) { $videoPath -replace "`"","" } else { "" }

    Write-Host "Booting Backend... (Video: '$cleanVideoPath', Resume: $resume)" -ForegroundColor Green
    
    $argList = @("`"$scriptPath`"", "--non-interactive")
    if ($resume) {
        $argList += "--resume"
    } elseif (-not [string]::IsNullOrWhiteSpace($cleanVideoPath)) {
        $argList += "--video"
        $argList += "`"$cleanVideoPath`""
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonExe
    $psi.WorkingDirectory = $scriptDir
    $psi.Arguments = $argList -join " "
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $global:pyProcess = [System.Diagnostics.Process]::Start($psi)
}

function Send-ResponseSafely ($response, [byte[]]$bytes, [string]$contentType = $null, [int]$statusCode = 200) {
    try {
        $response.StatusCode = $statusCode
        if ($contentType) { $response.ContentType = $contentType }
        if ($bytes) {
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        $response.Close()
    } catch [System.Net.HttpListenerException], [System.IO.IOException] {
        # Client aborted connection (e.g. browser poll timeout or refresh)
    } catch {
        Write-Host "Network response error: $_" -ForegroundColor DarkGray
    } finally {
        try { $response.Close() } catch {}
    }
}

try {
    while ($httpListener.IsListening) {
        try {
            $contextAsync = $httpListener.BeginGetContext($null, $null)
            while (!$contextAsync.AsyncWaitHandle.WaitOne(50)) {
                if (-not $httpListener.IsListening) { break }
            }
            if (-not $httpListener.IsListening) { break }

            $context = $httpListener.EndGetContext($contextAsync)
            $request = $context.Request
            $response = $context.Response

            $response.AddHeader("Access-Control-Allow-Origin", "*")
            $response.AddHeader("Access-Control-Allow-Headers", "Content-Type, X-File-Name")
            $url = $request.RawUrl

            if ($request.HttpMethod -eq "OPTIONS") {
                Send-ResponseSafely -response $response -bytes $null -statusCode 200
                continue
            }

            # 1. API: Process Status
            if ($url -eq "/api/status") {
                $isRunning = ($global:pyProcess -ne $null -and -not $global:pyProcess.HasExited)
                $resJson = @{ running = $isRunning; pid = if ($isRunning) { $global:pyProcess.Id } else { $null } } | ConvertTo-Json
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($resJson)
                Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                continue
            }

            # 2. API: Video File Ingestion
            if ($url -eq "/api/upload" -and $request.HttpMethod -eq "POST") {
                $disposition = $request.Headers["X-File-Name"]
                $fileName = if ($disposition) { [System.Uri]::UnescapeDataString($disposition) } else { "uploaded_video.mp4" }
                $targetPath = Join-Path $videosDir $fileName

                try {
                    $fs = New-Object System.IO.FileStream($targetPath, [System.IO.FileMode]::Create)
                    $request.InputStream.CopyTo($fs)
                    $fs.Close()

                    $resJson = @{ status = "ok"; path = $targetPath } | ConvertTo-Json
                    $bytes = [System.Text.Encoding]::UTF8.GetBytes($resJson)
                    Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                } catch {
                    Send-ResponseSafely -response $response -bytes $null -statusCode 500
                }
                continue
            }

            # 3. API: Actions (Start / Stop / Resume)
            if ($url -eq "/api/action" -and $request.HttpMethod -eq "POST") {
                try {
                    $reader = New-Object System.IO.StreamReader($request.InputStream, [System.Text.Encoding]::UTF8)
                    $body = $reader.ReadToEnd() | ConvertFrom-Json
                    
                    if ($body.action -eq "start") { Start-Backend -videoPath $body.videoPath -resume $false }
                    elseif ($body.action -eq "resume") { Start-Backend -videoPath $body.videoPath -resume $true }
                    elseif ($body.action -eq "stop") { Stop-Backend }

                    $bytes = [System.Text.Encoding]::UTF8.GetBytes('{"status":"ok"}')
                    Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                } catch {
                    Send-ResponseSafely -response $response -bytes $null -statusCode 500
                }
                continue
            }

            # 4. API: Current-Session Logs (Isolates output to the active session only)
            if ($url.StartsWith("/api/logs")) {
                $clientCursor = 0
                if ($request.QueryString["cursor"]) { 
                    [int]::TryParse($request.QueryString["cursor"], [ref]$clientCursor) | Out-Null 
                }

                $sessionLines = [System.Collections.Generic.List[string]]::new()

                if (Test-Path $logPath) {
                    try {
                        $fs = New-Object System.IO.FileStream($logPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
                        $rawAll = [System.Collections.Generic.List[string]]::new()
                        while (-not $sr.EndOfStream) {
                            $rawAll.Add($sr.ReadLine())
                        }
                        $sr.Close()
                        $fs.Close()

                        # Scan backwards to locate the start of the CURRENT session
                        $lastStartIdx = -1
                        for ($i = $rawAll.Count - 1; $i -ge 0; $i--) {
                            if ($rawAll[$i] -like "*=== Reconstruction Program Started ===*") {
                                $lastStartIdx = $i
                                break
                            }
                        }

                        if ($lastStartIdx -ge 0) {
                            for ($i = $lastStartIdx; $i -lt $rawAll.Count; $i++) {
                                $sessionLines.Add($rawAll[$i])
                            }
                        } else {
                            # Fallback if the marker hasn't been written yet
                            $sessionLines.AddRange($rawAll)
                        }
                    } catch {}
                }

                # Deliver only newly appended lines within the current session
                $newLines = @()
                $nextCursor = $sessionLines.Count
                if ($clientCursor -lt $sessionLines.Count) {
                    $newLines = $sessionLines.GetRange($clientCursor, ($sessionLines.Count - $clientCursor))
                }

                $resObj = [PSCustomObject]@{
                    nextCursor = $nextCursor
                    lines      = @($newLines)
                }
                $bytes = [System.Text.Encoding]::UTF8.GetBytes(($resObj | ConvertTo-Json -Compress))
                Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                continue
            }

            # 5. API: Scene Asset Discovery (Informs frontend what files exist)
            if ($url.StartsWith("/api/scene-assets")) {
                $sceneName = $request.QueryString["scene"]
                $targetDir = Join-Path $scriptDir "SCENES\$sceneName"
                $meshDir   = Join-Path $targetDir "mesh"

                $hasSplat = (Test-Path (Join-Path $meshDir "splat.ply"))
                if (-not $hasSplat) {
                    # Fallback check for any exported PLY
                    $hasSplat = ($null -ne (Get-ChildItem -Path $meshDir -Filter "*.ply" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "colored.ply" } | Select-Object -First 1))
                }

                $hasMesh = (Test-Path (Join-Path $meshDir "colored.obj"))

                $assets = @{
                    hasSplat = $hasSplat
                    hasMesh  = $hasMesh
                }

                $bytes = [System.Text.Encoding]::UTF8.GetBytes(($assets | ConvertTo-Json -Compress))
                Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                continue
            }

            # 6. API: Dedicated Model Loader (Delivers PLY or OBJ by explicit frontend request)
            if ($url.StartsWith("/api/model")) {
                $scene = $request.QueryString["scene"]
                $type  = $request.QueryString["type"] # 'splat' (PLY) or 'mesh' (OBJ)
                $meshDir = Join-Path $scriptDir "SCENES\$scene\mesh"
                $targetFile = $null
                $mimeType   = "application/octet-stream"

                if ($type -eq "splat") {
                    $candidate = Join-Path $meshDir "splat.ply"
                    if (Test-Path $candidate) {
                        $targetFile = $candidate
                    } else {
                        $found = Get-ChildItem -Path $meshDir -Filter "*.ply" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "colored.ply" } | Select-Object -First 1
                        if ($found) { $targetFile = $found.FullName }
                    }
                    $mimeType = "application/octet-stream"
                } elseif ($type -eq "mesh") {
                    $candidate = Join-Path $meshDir "colored.obj"
                    if (Test-Path $candidate) {
                        $targetFile = $candidate
                        $mimeType = "model/obj"
                    }
                }

                if ($targetFile -and (Test-Path $targetFile -PathType Leaf)) {
                    $bytes = [System.IO.File]::ReadAllBytes($targetFile)
                    Send-ResponseSafely -response $response -bytes $bytes -contentType $mimeType
                } else {
                    Send-ResponseSafely -response $response -bytes $null -statusCode 404
                }
                continue
            }

            # 7. API: Scenes Manifest
            if ($url -eq "/api/scenes") {
                $manifestContent = if (Test-Path $manifestPath) { [System.IO.File]::ReadAllText($manifestPath, [System.Text.Encoding]::UTF8) } else { "{}" }
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($manifestContent)
                Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                continue
            }

            # 8. API: Pipeline Config
            if ($url -eq "/api/config") {
                if ($request.HttpMethod -eq "GET") {
                    $cfgContent = if (Test-Path $configPath) { [System.IO.File]::ReadAllText($configPath, [System.Text.Encoding]::UTF8) } else { "{}" }
                    $bytes = [System.Text.Encoding]::UTF8.GetBytes($cfgContent)
                    Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                } elseif ($request.HttpMethod -eq "POST") {
                    try {
                        $reader = New-Object System.IO.StreamReader($request.InputStream, [System.Text.Encoding]::UTF8)
                        $parsed = $reader.ReadToEnd() | ConvertFrom-Json
                        [System.IO.File]::WriteAllText($configPath, ($parsed | ConvertTo-Json -Depth 5), [System.Text.Encoding]::UTF8)
                        $bytes = [System.Text.Encoding]::UTF8.GetBytes('{"status":"ok"}')
                        Send-ResponseSafely -response $response -bytes $bytes -contentType "application/json"
                    } catch {
                        Send-ResponseSafely -response $response -bytes $null -statusCode 500
                    }
                }
                continue
            }

            # 9. Serve Frontend Dashboard
            if ($url -eq "/" -or $url.StartsWith("/?")) {
                if (Test-Path $htmlPath) {
                    $htmlBytes = [System.IO.File]::ReadAllBytes($htmlPath)
                    Send-ResponseSafely -response $response -bytes $htmlBytes -contentType "text/html; charset=utf-8"
                } else {
                    Send-ResponseSafely -response $response -bytes $null -statusCode 404
                }
                continue
            }

            Send-ResponseSafely -response $response -bytes $null -statusCode 404
        } catch {
            Write-Host "HTTP Loop Exception: $_" -ForegroundColor DarkRed
        }
    }
}
finally {
    Stop-Backend
    $httpListener.Stop()
}