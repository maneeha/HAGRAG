# Data layout

The repository does not redistribute the PubMed article PDFs.

Place the paper resources in the following locations before running the full experiment:

```text
data/
├── pdfs/                              # selected PubMed PDFs
├── qa/
│   └── 100diabetes_qa_dataset.json    # curated PubMedQA evaluation set
└── corpus/
    └── HAGRAG_150_PDF_Corpus_List.csv # corpus manifest
```

The PDF directory is intentionally ignored by Git. The QA file and corpus manifest may be committed
when their redistribution terms permit it.
