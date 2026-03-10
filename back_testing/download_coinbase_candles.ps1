param(
    [string]$ProductId = "BTC-USD",
    [string]$Granularity = "FIVE_MINUTE",
    [string]$StartUtc = "2016-01-01T00:00:00Z",
    [string]$EndUtc = "2026-03-08T23:59:00Z",
    [string]$CredentialsPath = "D:\Quant\quant-lab\cdp_api_key.json",
    [string]$OutputPath = "D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\BTC_USD_FIVE_MINUTE_20160101_20260308.csv",
    [int]$Limit = 350,
    [int]$PauseMilliseconds = 125,
    [int]$MaxRetries = 5
)

$ErrorActionPreference = "Stop"

$granularitySeconds = @{
    ONE_MINUTE = 60
    FIVE_MINUTE = 300
    FIFTEEN_MINUTE = 900
    THIRTY_MINUTE = 1800
    ONE_HOUR = 3600
    TWO_HOUR = 7200
    FOUR_HOUR = 14400
    SIX_HOUR = 21600
    ONE_DAY = 86400
}

if (-not $granularitySeconds.ContainsKey($Granularity)) {
    throw "Unsupported granularity: $Granularity"
}

function ConvertTo-Base64Url {
    param([byte[]]$Bytes)

    return [Convert]::ToBase64String($Bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

function Convert-PemToDer {
    param([string]$PemText)

    $base64 = ($PemText -split "`r?`n" | Where-Object {
        $_ -and -not $_.StartsWith("-----BEGIN") -and -not $_.StartsWith("-----END")
    }) -join ""
    return [Convert]::FromBase64String($base64)
}

function Read-DerLength {
    param(
        [byte[]]$Bytes,
        [ref]$Offset
    )

    $firstByte = [int]$Bytes[$Offset.Value]
    $Offset.Value++
    if (($firstByte -band 0x80) -eq 0) {
        return $firstByte
    }

    $lengthByteCount = $firstByte -band 0x7F
    $length = 0
    for ($i = 0; $i -lt $lengthByteCount; $i++) {
        $length = ($length -shl 8) -bor [int]$Bytes[$Offset.Value]
        $Offset.Value++
    }
    return $length
}

function Read-DerValue {
    param(
        [byte[]]$Bytes,
        [ref]$Offset,
        [int]$ExpectedTag
    )

    if ([int]$Bytes[$Offset.Value] -ne $ExpectedTag) {
        throw "Unexpected DER tag. Expected $ExpectedTag, found $([int]$Bytes[$Offset.Value])."
    }
    $Offset.Value++
    $length = Read-DerLength -Bytes $Bytes -Offset $Offset
    $start = $Offset.Value
    $Offset.Value += $length
    return $Bytes[$start..($start + $length - 1)]
}

function Trim-LeadingZeros {
    param([byte[]]$Bytes)

    $index = 0
    while ($index -lt $Bytes.Length -and $Bytes[$index] -eq 0x00) {
        $index++
    }
    if ($index -ge $Bytes.Length) {
        return [byte[]](0)
    }
    return $Bytes[$index..($Bytes.Length - 1)]
}

function Import-EcPrivateKeyFromPem {
    param([string]$PemText)

    $der = Convert-PemToDer -PemText $PemText
    $offset = 0
    [void](Read-DerValue -Bytes $der -Offset ([ref]$offset) -ExpectedTag 0x30)

    $offset = 0
    if ([int]$der[$offset] -ne 0x30) {
        throw "Invalid EC private key: missing sequence."
    }
    $offset++
    [void](Read-DerLength -Bytes $der -Offset ([ref]$offset))
    [void](Read-DerValue -Bytes $der -Offset ([ref]$offset) -ExpectedTag 0x02)
    $privateKeyBytes = Read-DerValue -Bytes $der -Offset ([ref]$offset) -ExpectedTag 0x04

    $publicKeyBytes = $null
    while ($offset -lt $der.Length) {
        $tag = [int]$der[$offset]
        $offset++
        $contextLength = Read-DerLength -Bytes $der -Offset ([ref]$offset)
        $contextStart = $offset
        $offset += $contextLength
        $contextBytes = $der[$contextStart..($contextStart + $contextLength - 1)]

        if ($tag -eq 0xA1) {
            $innerOffset = 0
            $bitString = Read-DerValue -Bytes $contextBytes -Offset ([ref]$innerOffset) -ExpectedTag 0x03
            if ($bitString.Length -lt 2 -or $bitString[0] -ne 0x00) {
                throw "Unexpected EC public key bit string."
            }
            $publicKeyBytes = $bitString[1..($bitString.Length - 1)]
        }
    }

    if ($publicKeyBytes -eq $null -or $publicKeyBytes.Length -ne 65 -or $publicKeyBytes[0] -ne 0x04) {
        throw "Unable to parse uncompressed EC public key from PEM."
    }

    $ecParams = New-Object System.Security.Cryptography.ECParameters
    $ecParams.Curve = [System.Security.Cryptography.ECCurve]::CreateFromFriendlyName("nistP256")
    $ecParams.D = $privateKeyBytes
    $ecPoint = New-Object System.Security.Cryptography.ECPoint
    $ecPoint.X = [byte[]]$publicKeyBytes[1..32]
    $ecPoint.Y = [byte[]]$publicKeyBytes[33..64]
    $ecParams.Q = $ecPoint

    $ecdsa = [System.Security.Cryptography.ECDsa]::Create()
    $ecdsa.ImportParameters($ecParams)
    return $ecdsa
}

function Convert-DerSignatureToJose {
    param(
        [byte[]]$DerSignature,
        [int]$CoordinateLength = 32
    )

    if ($DerSignature.Length -eq ($CoordinateLength * 2)) {
        return $DerSignature
    }

    if ($DerSignature.Length -lt 8 -or $DerSignature[0] -ne 0x30) {
        throw "Unexpected DER signature format."
    }

    $offset = 2
    if ($DerSignature[$offset] -ne 0x02) {
        throw "Invalid DER signature: missing R marker."
    }

    $rLength = [int]$DerSignature[$offset + 1]
    $rBytes = $DerSignature[($offset + 2)..($offset + 1 + $rLength)]
    $offset = $offset + 2 + $rLength

    if ($DerSignature[$offset] -ne 0x02) {
        throw "Invalid DER signature: missing S marker."
    }

    $sLength = [int]$DerSignature[$offset + 1]
    $sBytes = $DerSignature[($offset + 2)..($offset + 1 + $sLength)]

    $rBytes = Trim-LeadingZeros -Bytes ([byte[]]$rBytes)
    $sBytes = Trim-LeadingZeros -Bytes ([byte[]]$sBytes)

    if ($rBytes.Length -gt $CoordinateLength -or $sBytes.Length -gt $CoordinateLength) {
        throw "DER signature coordinate length exceeded expected size."
    }

    $rawSignature = New-Object byte[] ($CoordinateLength * 2)
    [Array]::Copy($rBytes, 0, $rawSignature, $CoordinateLength - $rBytes.Length, $rBytes.Length)
    [Array]::Copy($sBytes, 0, $rawSignature, ($CoordinateLength * 2) - $sBytes.Length, $sBytes.Length)
    return $rawSignature
}

function New-CoinbaseJwt {
    param(
        [string]$Method,
        [string]$Path,
        [string]$KeyName,
        [System.Security.Cryptography.ECDsa]$PrivateKey
    )

    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $nonceBytes = New-Object byte[] 16
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($nonceBytes)
    $rng.Dispose()
    $nonce = ([BitConverter]::ToString($nonceBytes)).Replace("-", "").ToLowerInvariant()

    $headerJson = @{
        alg = "ES256"
        kid = $KeyName
        nonce = $nonce
        typ = "JWT"
    } | ConvertTo-Json -Compress

    $payloadJson = @{
        sub = $KeyName
        iss = "cdp"
        nbf = $now
        exp = $now + 120
        uri = "$Method api.coinbase.com$Path"
    } | ConvertTo-Json -Compress

    $encodedHeader = ConvertTo-Base64Url ([Text.Encoding]::UTF8.GetBytes($headerJson))
    $encodedPayload = ConvertTo-Base64Url ([Text.Encoding]::UTF8.GetBytes($payloadJson))
    $signingInput = "$encodedHeader.$encodedPayload"
    $signatureDer = $PrivateKey.SignData(
        [Text.Encoding]::UTF8.GetBytes($signingInput),
        [System.Security.Cryptography.HashAlgorithmName]::SHA256
    )
    $signatureJose = Convert-DerSignatureToJose -DerSignature $signatureDer
    $encodedSignature = ConvertTo-Base64Url $signatureJose
    return "$signingInput.$encodedSignature"
}

function Invoke-CoinbaseGet {
    param(
        [string]$Path,
        [hashtable]$Query,
        [string]$KeyName,
        [System.Security.Cryptography.ECDsa]$PrivateKey,
        [int]$MaxRetries
    )

    $queryString = ($Query.GetEnumerator() | Sort-Object Name | ForEach-Object {
        "{0}={1}" -f [System.Uri]::EscapeDataString($_.Key), [System.Uri]::EscapeDataString([string]$_.Value)
    }) -join "&"

    $attempt = 0
    while ($true) {
        try {
            $token = New-CoinbaseJwt -Method "GET" -Path $Path -KeyName $KeyName -PrivateKey $PrivateKey
            $headers = @{
                Authorization = "Bearer $token"
                Accept = "application/json"
            }
            $url = "https://api.coinbase.com$Path`?$queryString"
            return Invoke-RestMethod -Uri $url -Method Get -Headers $headers -TimeoutSec 60
        } catch {
            $attempt++
            if ($attempt -ge $MaxRetries) {
                throw
            }
            Start-Sleep -Seconds ([Math]::Pow(2, $attempt))
        }
    }
}

$credentials = Get-Content $CredentialsPath -Raw | ConvertFrom-Json
if (-not $credentials.name -or -not $credentials.privateKey) {
    throw "Credential file must contain 'name' and 'privateKey'."
}

$privateKeyPem = $credentials.privateKey
$ecdsa = Import-EcPrivateKeyFromPem -PemText $privateKeyPem

$startTimestamp = [DateTimeOffset]::Parse($StartUtc).ToUnixTimeSeconds()
$endTimestamp = [DateTimeOffset]::Parse($EndUtc).ToUnixTimeSeconds()
$intervalSeconds = $granularitySeconds[$Granularity]
$stepSeconds = $intervalSeconds * $Limit

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}
if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

$seenStarts = [System.Collections.Generic.HashSet[long]]::new()
# Coinbase candle responses behave as if the `start` bound is exclusive.
# Starting one interval earlier preserves the first requested candle.
$cursor = $startTimestamp - $intervalSeconds
$requestCount = 0
$rowCount = 0

while ($cursor -lt $endTimestamp) {
    $chunkEnd = [Math]::Min($endTimestamp, $cursor + $stepSeconds)
    $path = "/api/v3/brokerage/products/$ProductId/candles"
    $response = Invoke-CoinbaseGet `
        -Path $path `
        -Query @{
            start = [string]$cursor
            end = [string]$chunkEnd
            granularity = $Granularity
            limit = [string]$Limit
        } `
        -KeyName $credentials.name `
        -PrivateKey $ecdsa `
        -MaxRetries $MaxRetries

    $requestCount++
    $chunkRows = @()
    foreach ($candle in @($response.candles) | Sort-Object { [long]$_.start }) {
        $candleStart = [long]$candle.start
        if ($candleStart -lt $startTimestamp -or $candleStart -gt $endTimestamp) {
            continue
        }
        if (-not $seenStarts.Add($candleStart)) {
            continue
        }
        $timestampUtc = [DateTimeOffset]::FromUnixTimeSeconds($candleStart).UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
        $chunkRows += [PSCustomObject]@{
            product_id = $ProductId
            granularity = $Granularity
            timestamp_utc = $timestampUtc
            start_unix = $candleStart
            open = [double]$candle.open
            high = [double]$candle.high
            low = [double]$candle.low
            close = [double]$candle.close
            volume = [double]$candle.volume
        }
    }

    if ($chunkRows.Count -gt 0) {
        $rowCount += $chunkRows.Count
        if (Test-Path $OutputPath) {
            $chunkRows | Export-Csv -Path $OutputPath -NoTypeInformation -Encoding UTF8 -Append
        } else {
            $chunkRows | Export-Csv -Path $OutputPath -NoTypeInformation -Encoding UTF8
        }
    }

    if ($requestCount % 50 -eq 0) {
        Write-Host ("Requests: {0} | Rows: {1} | Cursor UTC: {2}" -f $requestCount, $rowCount, [DateTimeOffset]::FromUnixTimeSeconds($chunkEnd).UtcDateTime.ToString("u"))
    }

    $cursor = $chunkEnd
    Start-Sleep -Milliseconds $PauseMilliseconds
}

Write-Host ("Completed download: {0}" -f $OutputPath)
Write-Host ("Requests made: {0}" -f $requestCount)
Write-Host ("Rows written: {0}" -f $rowCount)
