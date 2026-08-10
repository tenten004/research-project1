param(
    [string]$OutputName = "XICT発表スライド_T123001_15分_20260810"
)

$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$here = if ($scriptPath) {
    Split-Path -Parent $scriptPath
}
else {
    Join-Path $PWD "research-project1\大学院入試"
}
$projectRoot = Split-Path -Parent $here
$confusionMatrixPath = Join-Path $projectRoot "mri-vit-classification\outputs\cv5_all_axial_3class\oof_detailed_analysis\confusion_matrices.png"
$outputPptx = Join-Path $here ($OutputName + ".pptx")
$outputPdf = Join-Path $here ($OutputName + ".pdf")
$previewDir = Join-Path $here ($OutputName + "_preview")

if (-not (Test-Path $confusionMatrixPath)) {
    throw "Confusion matrix image not found: $confusionMatrixPath"
}

function Get-Rgb([int]$red, [int]$green, [int]$blue) {
    return $red + (256 * $green) + (65536 * $blue)
}

$script:Color = @{
    Navy = Get-Rgb 31 60 89
    Blue = Get-Rgb 47 102 144
    LightBlue = Get-Rgb 222 235 247
    PaleBlue = Get-Rgb 239 245 250
    Gray = Get-Rgb 102 102 102
    MidGray = Get-Rgb 180 180 180
    LightGray = Get-Rgb 242 242 242
    Dark = Get-Rgb 36 36 36
    White = Get-Rgb 255 255 255
    Red = Get-Rgb 192 57 43
    PaleRed = Get-Rgb 252 235 233
    Green = Get-Rgb 46 125 50
    PaleGreen = Get-Rgb 232 245 233
    Gold = Get-Rgb 183 134 11
}

function Add-TextBox {
    param(
        $Slide,
        [string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize = 20,
        [bool]$Bold = $false,
        [int]$Color = $script:Color.Dark,
        [int]$Align = 1,
        [string]$FontName = "Yu Gothic",
        [int]$VerticalAnchor = 1
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.AutoSize = 0
    $shape.TextFrame2.VerticalAnchor = $VerticalAnchor
    $shape.TextFrame2.TextRange.Text = $Text.Replace("\n", "`n")
    $shape.TextFrame2.TextRange.Font.Name = $FontName
    $shape.TextFrame2.TextRange.Font.NameFarEast = $FontName
    $shape.TextFrame2.TextRange.Font.Size = $FontSize
    $shape.TextFrame2.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $Color
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Box {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [int]$Fill,
        [int]$Line = $script:Color.MidGray,
        [double]$RadiusStyle = 5
    )

    $shape = $Slide.Shapes.AddShape($RadiusStyle, $Left, $Top, $Width, $Height)
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Transparency = 0
    $shape.Line.ForeColor.RGB = $Line
    $shape.Line.Weight = 1
    return $shape
}

function Add-Line {
    param($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [int]$Color = $script:Color.MidGray, [double]$Weight = 1)
    $line = $Slide.Shapes.AddLine($X1, $Y1, $X2, $Y2)
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = $Weight
    return $line
}

function Add-SlideTitle {
    param($Slide, [string]$Title, [string]$Kicker = "")
    if ($Kicker) {
        Add-TextBox $Slide $Kicker 45 24 170 20 10 $true $script:Color.Blue | Out-Null
    }
    Add-TextBox $Slide $Title 45 42 870 45 27 $true $script:Color.Navy | Out-Null
    Add-Line $Slide 45 91 915 91 $script:Color.Blue 2.2 | Out-Null
}

function Add-Footer {
    param($Slide, [int]$Number, [string]$Timing, [string]$Source = "")
    if ($Source) {
        Add-TextBox $Slide $Source 45 505 760 16 8 $false $script:Color.Gray | Out-Null
    }
    Add-TextBox $Slide $Timing 805 504 80 16 8 $false $script:Color.Gray 3 | Out-Null
    Add-TextBox $Slide ([string]$Number) 895 502 20 18 9 $true $script:Color.Navy 3 | Out-Null
}

function Set-SpeakerNotes {
    param($Slide, [string]$Notes)
    foreach ($shape in $Slide.NotesPage.Shapes) {
        try {
            if ($shape.HasTextFrame -and $shape.PlaceholderFormat.Type -eq 2) {
                $shape.TextFrame.TextRange.Text = $Notes
                return
            }
        }
        catch {
            continue
        }
    }
}

function Add-NumberCard {
    param($Slide, [string]$Number, [string]$Label, [double]$Left, [double]$Top, [double]$Width, [int]$Accent = $script:Color.Blue)
    Add-Box $Slide $Left $Top $Width 105 $script:Color.White $script:Color.MidGray | Out-Null
    Add-TextBox $Slide $Number ($Left + 14) ($Top + 13) ($Width - 28) 45 28 $true $Accent 2 | Out-Null
    Add-TextBox $Slide $Label ($Left + 14) ($Top + 61) ($Width - 28) 32 12 $false $script:Color.Gray 2 | Out-Null
}

function Add-ProcessStep {
    param($Slide, [string]$Index, [string]$Title, [string]$Body, [double]$Left, [double]$Top, [double]$Width, [int]$Fill = $script:Color.PaleBlue)
    Add-Box $Slide $Left $Top $Width 110 $Fill $script:Color.MidGray | Out-Null
    $circle = $Slide.Shapes.AddShape(9, ($Left + 14), ($Top + 14), 30, 30)
    $circle.Fill.ForeColor.RGB = $script:Color.Blue
    $circle.Line.Visible = 0
    Add-TextBox $Slide $Index ($Left + 14) ($Top + 18) 30 22 12 $true $script:Color.White 2 | Out-Null
    Add-TextBox $Slide $Title ($Left + 54) ($Top + 14) ($Width - 68) 28 16 $true $script:Color.Navy | Out-Null
    Add-TextBox $Slide $Body ($Left + 16) ($Top + 52) ($Width - 32) 45 12 $false $script:Color.Dark | Out-Null
}

function Add-MetricRow {
    param($Slide, [string]$Metric, [string]$Cnn, [string]$Vit, [string]$Difference, [double]$Top, [bool]$Highlight = $false)
    $fill = if ($Highlight) { $script:Color.PaleGreen } else { $script:Color.White }
    $line = if ($Highlight) { $script:Color.Green } else { $script:Color.MidGray }
    Add-Box $Slide 70 $Top 820 48 $fill $line 1 | Out-Null
    Add-TextBox $Slide $Metric 88 ($Top + 12) 160 24 14 $Highlight $script:Color.Dark | Out-Null
    Add-TextBox $Slide $Cnn 270 ($Top + 12) 100 24 15 $false $script:Color.Gray 2 | Out-Null
    Add-TextBox $Slide $Vit 390 ($Top + 12) 100 24 15 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $Slide $Difference 520 ($Top + 12) 350 24 13 $Highlight $(if ($Highlight) { $script:Color.Green } else { $script:Color.Gray }) | Out-Null
}

if (Test-Path $outputPptx) { Remove-Item $outputPptx -Force }
if (Test-Path $outputPdf) { Remove-Item $outputPdf -Force }
if (Test-Path $previewDir) { Remove-Item $previewDir -Recurse -Force }
New-Item -ItemType Directory -Path $previewDir | Out-Null

$powerPoint = $null
$presentation = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    # 1. Title
    $slide = $presentation.Slides.Add(1, 12)
    $band = $slide.Shapes.AddShape(1, 0, 0, 960, 86)
    $band.Fill.ForeColor.RGB = $script:Color.Navy
    $band.Line.Visible = 0
    Add-TextBox $slide "X-ICT" 58 30 160 28 16 $true $script:Color.White | Out-Null
    Add-TextBox $slide "Vision Transformerを用いた\n大脳白質病変グレード分類の患者単位評価" 58 145 845 125 30 $true $script:Color.Navy | Out-Null
    Add-Line $slide 58 289 900 289 $script:Color.Blue 2.5 | Out-Null
    Add-TextBox $slide "浅川 天夢　／　石井 一夫" 58 315 500 32 18 $false $script:Color.Dark | Out-Null
    Add-TextBox $slide "公立諏訪東京理科大学 工学部 情報応用工学科" 58 354 620 28 14 $false $script:Color.Gray | Out-Null
    Add-TextBox $slide "患者重複のない5分割交差検証でViTとCNNを比較" 58 424 720 32 17 $true $script:Color.Blue | Out-Null
    Add-Footer $slide 1 "0:30"
    Set-SpeakerNotes $slide "これから、Vision Transformerを用いた大脳白質病変グレード分類の患者単位評価について発表します。本研究では、過去のCNN研究を出発点に、評価設計を患者単位へ見直し、ViTとCNNを公平に比較しました。"

    # 2. Background
    $slide = $presentation.Slides.Add(2, 12)
    Add-SlideTitle $slide "研究背景：白質病変を画像から客観評価する" "BACKGROUND"
    Add-Box $slide 55 120 265 300 $script:Color.PaleBlue $script:Color.LightBlue | Out-Null
    Add-TextBox $slide "大脳白質病変" 80 145 215 34 22 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "脳MRIのT2・FLAIRで\n高信号として観察" 80 205 215 62 17 $false $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "加齢・高血圧と関連\n脳卒中・認知機能低下の\nリスク指標" 80 292 215 86 16 $false $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "→ 重症度を再現性高く評価したい" 370 145 500 35 21 $true $script:Color.Blue | Out-Null
    Add-Box $slide 370 205 500 88 $script:Color.White $script:Color.MidGray | Out-Null
    Add-TextBox $slide "従来：CNN" 392 220 150 24 17 $true $script:Color.Navy | Out-Null
    Add-TextBox $slide "局所的な画像特徴の抽出に強い" 392 253 430 23 14 $false $script:Color.Dark | Out-Null
    Add-Box $slide 370 315 500 88 $script:Color.White $script:Color.MidGray | Out-Null
    Add-TextBox $slide "本研究：ViT" 392 330 150 24 17 $true $script:Color.Blue | Out-Null
    Add-TextBox $slide "Self-Attentionで広域的な関係を扱う" 392 363 430 23 14 $false $script:Color.Dark | Out-Null
    Add-TextBox $slide "ただし、モデル比較より先に評価単位を揃える必要がある" 155 453 650 28 16 $true $script:Color.Red 2 | Out-Null
    Add-Footer $slide 2 "1:10" "Debette & Markus, BMJ, 2010"
    Set-SpeakerNotes $slide "大脳白質病変は脳MRIで高信号として観察され、加齢や高血圧と関連します。進行は脳卒中や認知機能低下のリスクとも関連するため、画像から重症度を客観的に評価する意義があります。従来はCNNが用いられてきました。本研究では広域的な関係を扱えるViTに着目しましたが、公平な比較にはまず評価単位を患者で揃えることが重要です。"

    # 3. Research journey
    $slide = $presentation.Slides.Add(3, 12)
    Add-SlideTitle $slide "これまでの活動：再現から評価設計の見直しへ" "RESEARCH JOURNEY"
    Add-ProcessStep $slide "1" "先行研究を再現" "CNN・FLAIR＋T1・axial 9–15\n画像単位 accuracy 0.9209を参照" 50 125 205 $script:Color.LightGray
    Add-TextBox $slide "→" 260 162 35 28 22 $true $script:Color.Blue 2 | Out-Null
    Add-ProcessStep $slide "2" "分割を監査" "同一患者の別スライスが\n学習・評価に混在し得ると確認" 300 125 205 $script:Color.PaleRed
    Add-TextBox $slide "→" 510 162 35 28 22 $true $script:Color.Blue 2 | Out-Null
    Add-ProcessStep $slide "3" "課題を再定義" "患者単位split・3クラス化\n全アキシャルを利用" 550 125 175 $script:Color.PaleBlue
    Add-TextBox $slide "→" 730 162 35 28 22 $true $script:Color.Blue 2 | Out-Null
    Add-ProcessStep $slide "4" "最終比較" "同一5-fold・同一条件で\nViTとCNNをOOF評価" 770 125 145 $script:Color.PaleGreen
    Add-Box $slide 95 300 770 105 $script:Color.White $script:Color.Red | Out-Null
    Add-TextBox $slide "重要な転換" 120 320 140 27 18 $true $script:Color.Red | Out-Null
    Add-TextBox $slide "0.9209は画像単位の参考値。患者単位OOFの結果とは直接比較しない。" 120 357 700 28 17 $true $script:Color.Dark | Out-Null
    Add-TextBox $slide "過去発表の『全断面＋ViT』という方向性を、臨床利用に近い評価設計で検証した" 120 440 720 28 16 $true $script:Color.Blue 2 | Out-Null
    Add-Footer $slide 3 "1:30" "先行研究および本研究の実験記録に基づく"
    Set-SpeakerNotes $slide "研究の経過です。まず先行CNN研究の再現から始めました。先行結果の0.9209は画像単位の分割で得られた参考値です。データを監査すると、同じ患者の別スライスが学習と評価に混在し得ることが分かりました。そこで患者単位splitへ変更し、少数クラスを考慮してgrade 2から4を統合しました。最終的に、同一fold・同一学習条件でViTとCNNを比較しました。ここで0.9209と今回の患者単位結果は直接比較しません。"

    # 4. Aim
    $slide = $presentation.Slides.Add(4, 12)
    Add-SlideTitle $slide "研究目的" "OBJECTIVE"
    Add-TextBox $slide "患者重複のない条件で、ViTはCNNより有効か？" 100 125 760 52 26 $true $script:Color.Navy 2 | Out-Null
    Add-Box $slide 90 210 245 155 $script:Color.PaleBlue $script:Color.LightBlue | Out-Null
    Add-TextBox $slide "RQ1" 115 230 195 30 15 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "患者単位の\n分類性能" 115 273 195 60 21 $true $script:Color.Dark 2 | Out-Null
    Add-Box $slide 357 210 245 155 $script:Color.PaleBlue $script:Color.LightBlue | Out-Null
    Add-TextBox $slide "RQ2" 382 230 195 30 15 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "クラス別の\n識別傾向" 382 273 195 60 21 $true $script:Color.Dark 2 | Out-Null
    Add-Box $slide 625 210 245 155 $script:Color.PaleBlue $script:Color.LightBlue | Out-Null
    Add-TextBox $slide "RQ3" 650 230 195 30 15 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "改善量の\n不確実性" 650 273 195 60 21 $true $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "評価指標：Accuracy / macro-F1 / macro ROC-AUC / Balanced Accuracy / クラス別Recall" 100 417 760 50 15 $false $script:Color.Gray 2 | Out-Null
    Add-Footer $slide 4 "0:50"
    Set-SpeakerNotes $slide "研究目的は、患者重複のない条件でViTがCNNより有効かを検証することです。患者単位の分類性能だけでなく、各クラスの識別傾向と、改善量の不確実性も評価します。"

    # 5. Data
    $slide = $presentation.Slides.Add(5, 12)
    Add-SlideTitle $slide "対象データと分類課題" "DATA"
    Add-NumberCard $slide "1,154名" "脳ドック受診者" 60 122 190
    Add-NumberCard $slide "53,194枚" "全アキシャル画像" 265 122 190
    Add-NumberCard $slide "FLAIR＋T1" "最終比較の入力" 470 122 190
    Add-NumberCard $slide "3クラス" "患者単位ラベル" 675 122 190
    Add-TextBox $slide "クラス構成" 70 275 170 27 17 $true $script:Color.Navy | Out-Null
    $totalWidth = 780.0
    $g0 = $totalWidth * 557 / 1154
    $g1 = $totalWidth * 444 / 1154
    $g2 = $totalWidth - $g0 - $g1
    $bar = $slide.Shapes.AddShape(1, 70, 318, $g0, 62)
    $bar.Fill.ForeColor.RGB = $script:Color.Navy; $bar.Line.Visible = 0
    $bar = $slide.Shapes.AddShape(1, (70 + $g0), 318, $g1, 62)
    $bar.Fill.ForeColor.RGB = $script:Color.Blue; $bar.Line.Visible = 0
    $bar = $slide.Shapes.AddShape(1, (70 + $g0 + $g1), 318, $g2, 62)
    $bar.Fill.ForeColor.RGB = $script:Color.Gold; $bar.Line.Visible = 0
    Add-TextBox $slide "grade 0\n557名（48.3%）" 80 328 ($g0 - 20) 42 14 $true $script:Color.White 2 | Out-Null
    Add-TextBox $slide "grade 1\n444名（38.5%）" (80 + $g0) 328 ($g1 - 20) 42 14 $true $script:Color.White 2 | Out-Null
    Add-TextBox $slide "grade 2以上\n153名（13.3%）" (80 + $g0 + $g1) 328 ($g2 - 20) 42 12 $true $script:Color.White 2 | Out-Null
    Add-TextBox $slide "grade 3：39名、grade 4：8名と少ないため、grade 2–4を統合" 150 420 660 30 15 $true $script:Color.Red 2 | Out-Null
    Add-Footer $slide 5 "1:10" "Frozen protocol: grade0=557, grade1=444, grade2+=153"
    Set-SpeakerNotes $slide "対象は1,154名、FLAIRとT1の全アキシャル画像53,194枚です。ラベルは患者単位です。grade 3と4は症例数が非常に少ないため、grade 2から4を統合し、grade 0、grade 1、grade 2以上の3クラス分類としました。"

    # 6. Evaluation design
    $slide = $presentation.Slides.Add(6, 12)
    Add-SlideTitle $slide "評価設計：患者単位5-fold OOF" "PROTOCOL"
    Add-TextBox $slide "患者を層化して5分割" 70 120 260 30 18 $true $script:Color.Navy | Out-Null
    $foldColors = @($script:Color.Blue, $script:Color.LightBlue, $script:Color.LightBlue, $script:Color.LightBlue, $script:Color.LightBlue)
    for ($index = 0; $index -lt 5; $index++) {
        $x = 75 + ($index * 145)
        Add-Box $slide $x 170 125 65 $foldColors[$index] $script:Color.MidGray 1 | Out-Null
        $fontColor = if ($index -eq 0) { $script:Color.White } else { $script:Color.Navy }
        Add-TextBox $slide ("Fold " + ($index + 1)) ($x + 10) 190 105 24 14 $true $fontColor 2 | Out-Null
    }
    Add-TextBox $slide "学習：4 fold" 95 258 200 27 15 $false $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "検証：1 fold" 340 258 200 27 15 $false $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "× 5回" 610 258 180 27 15 $true $script:Color.Navy 2 | Out-Null
    Add-Line $slide 70 305 890 305 $script:Color.MidGray 1 | Out-Null
    Add-Box $slide 75 330 250 105 $script:Color.PaleGreen $script:Color.Green | Out-Null
    Add-TextBox $slide "患者リーケージ 0" 95 347 210 28 19 $true $script:Color.Green 2 | Out-Null
    Add-TextBox $slide "同一患者はfoldをまたがない" 95 387 210 24 12 $false $script:Color.Dark 2 | Out-Null
    Add-Box $slide 355 330 250 105 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "OOF予測 1,154名" 375 347 210 28 19 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "全患者が検証に1回だけ登場" 375 387 210 24 12 $false $script:Color.Dark 2 | Out-Null
    Add-Box $slide 635 330 250 105 $script:Color.LightGray $script:Color.MidGray | Out-Null
    Add-TextBox $slide "paired bootstrap" 655 347 210 28 19 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "10,000回・95%信頼区間" 655 387 210 24 12 $false $script:Color.Dark 2 | Out-Null
    Add-Footer $slide 6 "1:30" "StratifiedKFold, random_state=1"
    Set-SpeakerNotes $slide "評価は患者単位の層化5分割交差検証です。同じ患者が学習と検証に重ならないようにし、全患者が検証に1回だけ現れるout-of-fold予測を得ました。これにより単一splitへの依存を抑えます。モデル間の差は同じ患者の予測を対応させたbootstrapを1万回行い、95パーセント信頼区間で評価しました。"

    # 7. Models and aggregation
    $slide = $presentation.Slides.Add(7, 12)
    Add-SlideTitle $slide "比較モデルと患者単位集約" "METHOD"
    Add-Box $slide 60 120 255 145 $script:Color.LightGray $script:Color.MidGray | Out-Null
    Add-TextBox $slide "CNN" 85 142 205 28 16 $true $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "ResNet18" 85 185 205 42 27 $true $script:Color.Dark 2 | Out-Null
    Add-Box $slide 645 120 255 145 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "ViT" 670 142 205 28 16 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "DeiT-small\npatch16 / 224" 670 180 205 62 23 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "同一条件" 395 140 170 28 17 $true $script:Color.Red 2 | Out-Null
    Add-TextBox $slide "224×224\n30 epoch\nAdamW / 3×10⁻⁵\n同一augmentation" 395 178 170 91 14 $false $script:Color.Dark 2 | Out-Null
    Add-Line $slide 190 285 770 285 $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "各患者の全スライスを推論" 80 316 230 34 17 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "→" 320 320 45 25 20 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "確信度上位5枚を選択" 365 316 230 34 17 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "→" 605 320 45 25 20 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "確率を平均して患者予測" 650 316 230 34 17 $true $script:Color.Navy 2 | Out-Null
    Add-Box $slide 120 390 720 68 $script:Color.PaleGreen $script:Color.Green | Out-Null
    Add-TextBox $slide "全断面を入力しつつ、手動のaxial範囲指定を避ける" 145 410 670 28 17 $true $script:Color.Green 2 | Out-Null
    Add-Footer $slide 7 "1:15" "Best validation-loss checkpoint / top-5 confidence pooling"
    Set-SpeakerNotes $slide "比較するのはResNet18とDeiT-smallです。入力サイズ、fold、epoch数、optimizer、データ拡張を揃えました。各患者の全スライスを推論し、確信度が高い上位5枚のクラス確率を平均して患者予測とします。全断面を利用しながら、手動でaxial範囲を限定しない設計です。"

    # 8. Main results
    $slide = $presentation.Slides.Add(8, 12)
    Add-SlideTitle $slide "主結果：ViTはmacro ROC-AUCを改善" "RESULTS"
    Add-TextBox $slide "指標" 88 111 160 22 11 $true $script:Color.Gray | Out-Null
    Add-TextBox $slide "CNN" 270 111 100 22 11 $true $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "ViT" 390 111 100 22 11 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "ViT − CNN（95% CI）" 520 111 350 22 11 $true $script:Color.Gray | Out-Null
    Add-MetricRow $slide "Accuracy" "0.6282" "0.6352" "+0.0069（−0.0156, 0.0295）" 140
    Add-MetricRow $slide "Macro-F1" "0.6001" "0.6134" "+0.0133（−0.0135, 0.0399）" 195
    Add-MetricRow $slide "Macro ROC-AUC" "0.7891" "0.8100" "+0.0209（0.0062, 0.0357）" 250 $true
    Add-MetricRow $slide "Balanced Acc." "0.5995" "0.6132" "+0.0137（−0.0112, 0.0386）" 305
    Add-Box $slide 95 390 770 73 $script:Color.PaleGreen $script:Color.Green | Out-Null
    Add-TextBox $slide "95% CIが0をまたがなかったのはmacro ROC-AUCのみ" 120 409 720 30 19 $true $script:Color.Green 2 | Out-Null
    Add-TextBox $slide "AccuracyのMcNemar検定：p = 0.598" 120 442 720 18 11 $false $script:Color.Gray 2 | Out-Null
    Add-Footer $slide 8 "1:50" "患者単位5-fold OOF、paired bootstrap 10,000回"
    Set-SpeakerNotes $slide "主結果です。ViTは4指標すべてでCNNを上回りました。特にmacro ROC-AUCはCNNの0.7891に対してViTは0.8100で、差は0.0209でした。95パーセント信頼区間は0.0062から0.0357で、0を含みませんでした。一方、Accuracy、macro-F1、Balanced Accuracyの差は信頼区間が0をまたぎました。したがって、ViTは患者の重症度を順位づける能力を改善したといえますが、分類性能全般で優れるとは結論できません。"

    # 9. Confusion matrices
    $slide = $presentation.Slides.Add(9, 12)
    Add-SlideTitle $slide "クラス別結果：grade 1の識別が共通課題" "CLASS-WISE RESULTS"
    $slide.Shapes.AddPicture($confusionMatrixPath, 0, -1, 65, 112, 830, 341) | Out-Null
    Add-Box $slide 95 458 770 36 $script:Color.PaleRed $script:Color.Red | Out-Null
    Add-TextBox $slide "grade 1 Recall：CNN 34.7% → ViT 38.5%（差の95% CIは0を含む）" 120 466 720 22 14 $true $script:Color.Red 2 | Out-Null
    Add-Footer $slide 9 "1:15" "行正規化混同行列、n=1,154"
    Set-SpeakerNotes $slide "混同行列を見ると、両モデルともgrade 0は比較的よく分類できています。grade 2以上のRecallはCNN 58.8パーセント、ViT 61.4パーセントです。一方、grade 1はCNN 34.7パーセント、ViT 38.5パーセントにとどまり、多くがgrade 0へ誤分類されました。ViTで改善傾向はありますが、差の信頼区間は0を含みます。"

    # 10. Error audit
    $slide = $presentation.Slides.Add(10, 12)
    Add-SlideTitle $slide "誤分類監査：grade 1の境界性を検討" "ERROR ANALYSIS"
    Add-NumberCard $slide "444名" "grade 1の監査対象" 70 120 190 $script:Color.Navy
    Add-NumberCard $slide "233名" "両モデルとも誤分類" 280 120 190 $script:Color.Red
    Add-NumberCard $slide "196名" "両モデルともgrade 0へ" 490 120 190 $script:Color.Red
    Add-NumberCard $slide "3名" "技術的フラグあり" 700 120 190 $script:Color.Green
    Add-TextBox $slide "示唆" 80 280 100 25 17 $true $script:Color.Navy | Out-Null
    Add-TextBox $slide "・単純なデータ欠損だけでは、grade 1の低Recallを説明できない\n・grade 0 / 1の境界、ラベルの揺らぎ、選択スライスの妥当性を確認する必要\n・高確信度誤分類を含む200例の盲検レビュー用パッケージを作成済み" 100 322 760 102 17 $false $script:Color.Dark | Out-Null
    Add-Box $slide 155 446 650 40 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "技術監査の次は、医療者による独立した再評価へ" 180 455 600 23 16 $true $script:Color.Blue 2 | Out-Null
    Add-Footer $slide 10 "1:15" "Grade1 OOF technical audit（医療的再判定は未実施）"
    Set-SpeakerNotes $slide "grade 1の誤分類を技術的に監査しました。444名のうち233名は両モデルとも誤分類し、そのうち196名は両方ともgrade 0へ分類しました。一方、画像枚数不足などの技術的フラグは3名だけでした。したがって単純なデータ欠損よりも、grade 0と1の境界性、ラベルの揺らぎ、あるいはtop-5で適切なスライスを拾えているかが課題と考えられます。医療者による盲検レビューは今後実施します。"

    # 11. Interpretation and limitations
    $slide = $presentation.Slides.Add(11, 12)
    Add-SlideTitle $slide "考察・限界・次の一手" "DISCUSSION"
    Add-Box $slide 55 120 270 300 $script:Color.PaleGreen $script:Color.Green | Out-Null
    Add-TextBox $slide "今回いえること" 80 143 220 30 18 $true $script:Color.Green 2 | Out-Null
    Add-TextBox $slide "ViTはmacro ROC-AUCと\n確率の較正指標を改善\n\nただしAccuracy・macro-F1の\n優位性は未確定" 80 200 220 146 17 $false $script:Color.Dark 2 | Out-Null
    Add-Box $slide 345 120 270 300 $script:Color.PaleRed $script:Color.Red | Out-Null
    Add-TextBox $slide "限界" 370 143 220 30 18 $true $script:Color.Red 2 | Out-Null
    Add-TextBox $slide "・単施設データ\n・探索後に固定したprotocol\n・top-5は後処理\n・患者内の断面間関係を\n　直接学習していない" 375 198 210 155 16 $false $script:Color.Dark | Out-Null
    Add-Box $slide 635 120 270 300 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "次の一手" 660 143 220 30 18 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "・外部検証／反復CV\n・grade 1の盲検再評価\n・MILやTransformerで\n　全スライスを患者単位学習" 665 198 210 155 16 $false $script:Color.Dark | Out-Null
    Add-TextBox $slide "『高い正解率』よりも、『誰に対してどこまで一般化するか』を重視" 120 453 720 30 18 $true $script:Color.Navy 2 | Out-Null
    Add-Footer $slide 11 "1:15"
    Set-SpeakerNotes $slide "今回、ViTはmacro ROC-AUCを改善し、詳細解析ではNLLやBrier scoreも改善しました。ただしAccuracyとmacro-F1の優位性は未確定です。限界として単施設データであること、探索後にprotocolを固定したこと、top-5が後処理で患者内の断面関係を直接学習していないことがあります。今後は外部検証、grade 1の盲検再評価、そしてMILやTransformerによる患者単位学習へ進めます。"

    # 12. Conclusion
    $slide = $presentation.Slides.Add(12, 12)
    Add-SlideTitle $slide "まとめ" "CONCLUSION"
    Add-Box $slide 90 125 780 86 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "1" 115 148 42 36 24 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "患者重複のない5-fold OOFでViTとCNNを比較した" 175 148 650 34 20 $true $script:Color.Navy | Out-Null
    Add-Box $slide 90 235 780 86 $script:Color.PaleGreen $script:Color.Green | Out-Null
    Add-TextBox $slide "2" 115 258 42 36 24 $true $script:Color.Green 2 | Out-Null
    Add-TextBox $slide "ViTはmacro ROC-AUCを0.0209改善した" 175 258 650 34 20 $true $script:Color.Navy | Out-Null
    Add-Box $slide 90 345 780 86 $script:Color.PaleRed $script:Color.Red | Out-Null
    Add-TextBox $slide "3" 115 368 42 36 24 $true $script:Color.Red 2 | Out-Null
    Add-TextBox $slide "grade 1の識別と外部検証が次の課題である" 175 368 650 34 20 $true $script:Color.Navy | Out-Null
    Add-TextBox $slide "ご清聴ありがとうございました" 205 463 550 32 20 $true $script:Color.Blue 2 | Out-Null
    Add-Footer $slide 12 "0:40"
    Set-SpeakerNotes $slide "まとめです。患者重複のない5-fold OOF評価でViTとCNNを比較しました。ViTはmacro ROC-AUCを0.0209改善しましたが、Accuracyとmacro-F1の優位性は確認できませんでした。grade 1の識別改善と外部検証が次の課題です。ご清聴ありがとうございました。"

    # 13. Appendix: references
    $slide = $presentation.Slides.Add(13, 12)
    Add-SlideTitle $slide "参考文献" "APPENDIX"
    Add-TextBox $slide "[1] 竹村典晃『畳み込みニューラルネットワークを用いた大脳白質病変のグレード予測』公立諏訪東京理科大学卒業論文, 2023.\n\n[2] S. Debette and H. S. Markus, The clinical importance of white matter hyperintensities on brain magnetic resonance imaging, BMJ, 341:c3666, 2010.\n\n[3] A. Dosovitskiy et al., An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale, ICLR, 2021.\n\n[4] H. Touvron et al., Training data-efficient image transformers and distillation through attention, ICML, 2021." 85 130 790 270 16 $false $script:Color.Dark | Out-Null
    Add-TextBox $slide "以降は質疑用スライド" 300 455 360 25 15 $true $script:Color.Gray 2 | Out-Null
    Add-Footer $slide 13 "参考"

    # 14. Appendix: original grades
    $slide = $presentation.Slides.Add(14, 12)
    Add-SlideTitle $slide "元grade別の検出結果" "APPENDIX"
    Add-TextBox $slide "元grade" 90 120 120 22 12 $true $script:Color.Gray | Out-Null
    Add-TextBox $slide "患者数" 250 120 100 22 12 $true $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "CNN Recall" 400 120 130 22 12 $true $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "ViT Recall" 580 120 130 22 12 $true $script:Color.Blue 2 | Out-Null
    $gradeRows = @(
        @("grade 0", "557", "0.864", "0.840"),
        @("grade 1", "444", "0.347", "0.385"),
        @("grade 2", "106", "0.481", "0.500"),
        @("grade 3", "39", "0.795", "0.846"),
        @("grade 4", "8", "1.000", "1.000")
    )
    $rowTop = 155
    foreach ($row in $gradeRows) {
        Add-Box $slide 80 $rowTop 720 48 $script:Color.White $script:Color.MidGray 1 | Out-Null
        Add-TextBox $slide $row[0] 100 ($rowTop + 12) 120 22 15 $true $script:Color.Dark | Out-Null
        Add-TextBox $slide $row[1] 250 ($rowTop + 12) 100 22 15 $false $script:Color.Dark 2 | Out-Null
        Add-TextBox $slide $row[2] 400 ($rowTop + 12) 130 22 15 $false $script:Color.Gray 2 | Out-Null
        Add-TextBox $slide $row[3] 580 ($rowTop + 12) 130 22 15 $true $script:Color.Blue 2 | Out-Null
        $rowTop += 55
    }
    Add-TextBox $slide "grade 4は8名のみであり、1.000という値の解釈には注意が必要" 135 455 690 26 15 $true $script:Color.Red 2 | Out-Null
    Add-Footer $slide 14 "参考" "3クラス学習後の元grade別Recall"

    # 15. Appendix: comparison caveat
    $slide = $presentation.Slides.Add(15, 12)
    Add-SlideTitle $slide "なぜ0.9209と今回のAccuracyを比較しないのか" "APPENDIX"
    Add-Box $slide 75 130 340 250 $script:Color.LightGray $script:Color.MidGray | Out-Null
    Add-TextBox $slide "先行研究の参考値" 100 155 290 30 18 $true $script:Color.Gray 2 | Out-Null
    Add-TextBox $slide "Accuracy 0.9209" 100 212 290 45 28 $true $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "画像単位split\n5クラス\naxial 9–15" 100 285 290 75 16 $false $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "≠" 447 225 66 45 30 $true $script:Color.Red 2 | Out-Null
    Add-Box $slide 545 130 340 250 $script:Color.PaleBlue $script:Color.Blue | Out-Null
    Add-TextBox $slide "本研究の最終評価" 570 155 290 30 18 $true $script:Color.Blue 2 | Out-Null
    Add-TextBox $slide "Accuracy 0.6352" 570 212 290 45 28 $true $script:Color.Navy 2 | Out-Null
    Add-TextBox $slide "患者単位5-fold OOF\n3クラス\n全アキシャル" 570 285 290 75 16 $false $script:Color.Dark 2 | Out-Null
    Add-TextBox $slide "評価単位・クラス定義・断面範囲が異なるため、性能低下とは断定できない" 115 425 730 35 17 $true $script:Color.Red 2 | Out-Null
    Add-Footer $slide 15 "参考"

    $presentation.SaveAs($outputPptx, 24)
    $presentation.SaveAs($outputPdf, 32)
    $presentation.Export($previewDir, "PNG", 1280, 720)

    Write-Output "Saved PPTX: $outputPptx"
    Write-Output "Saved PDF:  $outputPdf"
    Write-Output "Slides:     $($presentation.Slides.Count)"
    Write-Output "Preview:    $previewDir"
}
finally {
    if ($presentation) {
        $presentation.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    }
    if ($powerPoint) {
        $powerPoint.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}