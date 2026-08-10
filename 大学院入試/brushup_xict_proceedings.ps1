$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$here = if ($scriptPath) {
    Split-Path -Parent $scriptPath
}
else {
    Join-Path $PWD "research-project1\大学院入試"
}

$sourcePath = Join-Path $here "XICT予稿_T123001_浅川天夢_研究成果_20260804.docx"
$outputDocx = Join-Path $here "XICT予稿_T123001_浅川天夢_ブラッシュアップ版_20260810.docx"

if (-not (Test-Path $sourcePath)) {
    throw "Source document not found: $sourcePath"
}

$updates = @(
    @{
        Match = "Vision Transformerを用いた大脳白質病変グレード分類の患者単位評価"
        Text = "Vision Transformerを用いた大脳白質病変グレード分類性能の検討"
        Exact = $true
    },
    @{
        Match = "概要:"
        Text = "概要: 本研究の目的は，脳MRI画像を用いた大脳白質病変グレード分類におけるVision Transformer（ViT）の分類性能を検討することである．FLAIR・T1強調画像の全アキシャル53,194枚，1,154名をgrade 0，grade 1，grade 2以上の3クラスに統合し，患者重複のない5分割交差検証によりCNNと比較した．患者単位のout-of-fold評価では，ViTのmacro ROC-AUCは0.8100，CNNは0.7891で，差0.0209の95%信頼区間は0.0062–0.0357であった．一方，Accuracyとmacro-F1の差は明確でなかった．"
        Exact = $false
    },
    @{
        Match = "Patient-Level Evaluation of Cerebral White Matter Lesion Grade Classification Using Vision Transformer"
        Text = "Evaluation of Vision Transformer Performance for Cerebral White Matter Lesion Grade Classification"
        Exact = $true
    },
    @{
        Match = "Abstract:"
        Text = "Abstract: The objective of this study was to evaluate the classification performance of a Vision Transformer (ViT) for cerebral white matter lesion grades from brain MRI. We analyzed 53,194 FLAIR and T1-weighted axial images from 1,154 patients in three classes and performed leakage-free five-fold cross-validation with a CNN comparator. Patient-level out-of-fold macro ROC-AUC was 0.8100 for ViT and 0.7891 for CNN; the paired difference was 0.0209 (95% CI: 0.0062–0.0357). Differences in accuracy and macro-F1 were inconclusive."
        Exact = $false
    },
    @{
        Match = "先行研究ではCNN"
        Text = "先行研究ではCNNを用いたgrade 0からgrade 4の分類が行われた[1]．提案発表時には画像単位分割によるAccuracy 0.9209を比較基準としていたが，患者IDを再点検すると，同一患者の別スライスが学習集合と評価集合に重複する構造が判明した．そこで本研究では，評価単位を患者に改め，患者重複のない条件でVision Transformer（ViT）[3]を用いた大脳白質病変の分類性能を検討する．性能を相対的に評価するため，同一条件のCNNを比較モデルとする．"
        Exact = $false
    },
    @{
        Match = "2.2 比較条件と患者集約"
        Text = "2.2 モデルと患者単位評価"
        Exact = $true
    },
    @{
        Match = "患者単位の分割へ変更した結果"
        Text = "患者単位OOF評価において，ViTはmacro ROC-AUC 0.8100を示し，同一条件のCNNより0.0209高かった．Self-Attentionによる広域特徴の利用が，患者の重症度を順位づける能力に寄与した可能性がある．ただし，Accuracyとmacro-F1の差は明確でなく，ViTが分類性能全般でCNNより優れるとは結論できない．"
        Exact = $false
    },
    @{
        Match = "本研究は全アキシャルを入力し"
        Text = "画像単位分割による先行値と本研究は評価単位やクラス定義が異なるため，性能を直接比較できない．本研究は患者リーケージを防ぎ，未知患者に対する分類性能を評価した．また，全アキシャルを入力し，固定したtop-5集約で患者予測を得たため，手動の断面範囲指定を避けたが，スライス選択は確信度に基づく後処理であり，患者内の断面間関係を直接学習していない．また，本プロトコルは探索用validationでの検討後に固定したため，外部データに対する完全な未検証評価ではない．"
        Exact = $false
    },
    @{
        Match = "大脳白質病変の3クラス分類について"
        Text = "本研究では，大脳白質病変の3クラス分類におけるViTの分類性能を，患者重複のない5-fold OOF評価で検討した．ViTはmacro ROC-AUC 0.8100を示し，同一条件のCNNに対して小さいが統計的に支持される改善を示した一方，Accuracyとmacro-F1の優位性は確認できなかった．今後は外部検証または反復交差検証を行うとともに，全スライスを患者単位で学習するMILやTransformer集約へ発展させ，grade 1の識別改善を検討する．"
        Exact = $false
    }
)

Copy-Item -LiteralPath $sourcePath -Destination $outputDocx -Force

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $document = $word.Documents.Open($outputDocx, $false, $false)
    $changed = 0

    foreach ($paragraph in $document.Paragraphs) {
        $current = ($paragraph.Range.Text -replace "[\r\a]+$", "").Trim()
        foreach ($update in $updates) {
            $matches = if ($update.Exact) {
                $current -eq $update.Match
            }
            else {
                $current.StartsWith($update.Match)
            }

            if ($matches) {
                $range = $paragraph.Range.Duplicate
                $range.End = $range.End - 1
                $range.Text = $update.Text
                $changed++
                break
            }
        }
    }

    if ($changed -ne $updates.Count) {
        throw "Expected $($updates.Count) replacements, changed $changed"
    }

    $document.Save()
    $pages = $document.ComputeStatistics(2)

    Write-Output "Changed: $changed"
    Write-Output "Pages: $pages"
    Write-Output "Saved DOCX: $outputDocx"
}
finally {
    if ($document) {
        $document.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}