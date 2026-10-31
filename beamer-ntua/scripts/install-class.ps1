$ErrorActionPreference = "Stop"

$repoSlug = if ($env:NTUA_BEAMER_REPO) { $env:NTUA_BEAMER_REPO } else { "constatza/beamer-ntua" }
$repoRef = if ($env:NTUA_BEAMER_REF) { $env:NTUA_BEAMER_REF } else { "main" }
$baseUrl = if ($env:NTUA_BEAMER_BASE_URL) { $env:NTUA_BEAMER_BASE_URL } else { "https://raw.githubusercontent.com/$repoSlug/$repoRef" }

function Resolve-TexmfHome {
  if ($env:TEXMFHOME) {
    return $env:TEXMFHOME
  }

  $kpsewhich = Get-Command kpsewhich -ErrorAction SilentlyContinue
  if (-not $kpsewhich) {
    throw "kpsewhich is required unless TEXMFHOME is already set."
  }

  $value = & kpsewhich -var-value=TEXMFHOME
  if (-not $value) {
    throw "Could not resolve TEXMFHOME via kpsewhich."
  }

  return $value.Trim()
}

function Download-File {
  param(
    [string]$Url,
    [string]$Target
  )

  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Target
}

function Refresh-TexDb {
  $mktexlsr = Get-Command mktexlsr -ErrorAction SilentlyContinue
  if ($mktexlsr) {
    & mktexlsr *> $null
    return
  }

  $initexmf = Get-Command initexmf -ErrorAction SilentlyContinue
  if ($initexmf) {
    & initexmf --update-fndb *> $null
  }
}

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ntuabeamer-install-" + [System.Guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $tmpDir
$assetsDir = Join-Path $tmpDir "assets"
$null = New-Item -ItemType Directory -Path $assetsDir

try {
  foreach ($file in @("ntuabeamer.cls", "ntuabeamer.sty", "macros.tex", "theme.tex")) {
    Download-File -Url "$baseUrl/packages/ntuabeamer-class/framework/$file" -Target (Join-Path $tmpDir $file)
  }

  foreach ($asset in @("mgroup.png", "ntua.png", "school-civil-engineering.jpg")) {
    Download-File -Url "$baseUrl/packages/ntuabeamer-class/framework/assets/$asset" -Target (Join-Path $assetsDir $asset)
  }

  $texmfhome = Resolve-TexmfHome
  $installDir = Join-Path $texmfhome "tex/latex/ntuabeamer"

  if (Test-Path $installDir) {
    Remove-Item -Recurse -Force $installDir
  }

  $null = New-Item -ItemType Directory -Path $installDir

  Copy-Item (Join-Path $tmpDir "ntuabeamer.cls") $installDir
  Copy-Item (Join-Path $tmpDir "ntuabeamer.sty") $installDir
  Copy-Item (Join-Path $tmpDir "macros.tex") $installDir
  Copy-Item (Join-Path $tmpDir "theme.tex") $installDir
  Copy-Item (Join-Path $assetsDir "*") $installDir

  Refresh-TexDb

  Write-Host "Installed class -> $installDir"
}
finally {
  if (Test-Path $tmpDir) {
    Remove-Item -Recurse -Force $tmpDir
  }
}
