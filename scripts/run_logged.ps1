param(
  [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
  [string[]]$Command
)

$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs/$ts.log"

"=== $(Get-Date -Format o) ===" | Tee-Object -FilePath $log
"CMD: $($Command -join ' ')" | Tee-Object -FilePath $log -Append
"" | Tee-Object -FilePath $log -Append

& $Command[0] $Command[1..($Command.Length-1)] 2>&1 | Tee-Object -FilePath $log -Append

"" | Tee-Object -FilePath $log -Append
"=== exit $LASTEXITCODE ===" | Tee-Object -FilePath $log -Append
Write-Host "`nSaved: $log" -ForegroundColor Cyan
