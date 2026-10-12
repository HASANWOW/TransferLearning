# Tugas 05 — Transfer Learning pada Oxford-IIIT Pet (10 kelas)

Perbandingan from scratch (B1), feature extraction (B2), progressive fine-tuning (B3), uji kontrol augmentasi (B4),
dan arsitektur efisien EfficientNet-B0 / MobileNetV3-Small (B5) dengan ResNet-18 sebagai arsitektur utama.

## Struktur
```
TransferLearning.ipynb           # notebook utama (sudah dieksekusi, luaran tersimpan)
Laporan.pdf                      # laporan analisis (5 halaman)
requirements.txt
metrics_per_epoch.csv            # log mentah per epoch semua eksperimen
summary.csv / results.json       # ringkasan metrik akhir + konfigurasi & environment
summary_table.md                 # tabel konsolidasi (C1)
split_manifest.csv               # daftar berkas per split (train/val/test)
misclassified_best_model.csv     # sampel misklasifikasi (C3)
*.png                            # seluruh grafik (C1–C5)
```

## Menjalankan ulang
1. Buka notebook di Google Colab → `Runtime → Change runtime type → T4 GPU`.
2. `Runtime → Run all` (±30–50 menit). Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet`.
3. Lokal: `pip install -r requirements.txt` lalu jalankan notebook dengan Jupyter.

## Protokol
ResNet-18 `IMAGENET1K_V1` · 224×224 · normalisasi ImageNet · batch 32 · AdamW (wd 1e-2) · CrossEntropyLoss ·
early stopping pada val loss (patience 5, maks 30 epoch) · seed 42 · split stratified 70/15/15.

| Kode | Strategi | LR |
|---|---|---|
| B1 | from scratch, semua layer | 1e-3 |
| B2 | backbone beku, latih `fc` | fc 1e-3 |
| B3 | unfreeze `layer4` + `fc` | layer4 1e-4, fc 1e-3 |
| B4 | B3 tanpa vs dengan augmentasi | sama dengan B3 |
| B5 | strategi terbaik pada EfficientNet-B0 & MobileNetV3-Small | tail 1e-4, head 1e-3 |

## Hasil (test set, 298 citra)
| Kode | Strategi | Trainable params | Test acc | Test Macro F1 |
|---|---|---|---|---|
| B1 | ResNet-18 from scratch | 11.181.642 | 0,440 | 0,416 |
| B2 | ResNet-18 feature extraction | 5.130 | 0,873 | 0,871 |
| B3 | ResNet-18 fine-tune layer4 + fc | 8.398.858 | 0,916 | 0,915 |
| B4 | B3 tanpa augmentasi | 8.398.858 | 0,923 | 0,921 |
| B5 | EfficientNet-B0 fine-tune | 1.142.202 | 0,930 | 0,929 |
| B5 | MobileNetV3-Small fine-tune | 1.337.586 | 0,896 | 0,896 |

Detail lengkap ada di `outputs/summary.csv` dan laporan PDF.

Penulis: HASAN — 2802529203
