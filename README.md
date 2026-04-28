# Quranic Citation Transformer

Extracts Quranic citations from tafsir text files and predicts their sura/aya references using an LLM.

## Overview

This tool processes plain text files containing Quranic citations wrapped in curly braces (`{...}`), stores them in a PostgreSQL database, and uses the Mistral API to identify the corresponding sura and aya numbers.

The workflow has two stages:

1. **Insert** — parse citations from files in `input/` and upsert them into the database.
2. **Predict** — for each citation without a prediction, query the LLM and store the result.

## Requirements

- Python 3.12+
- PostgreSQL with a `citations` table in the `quranic_citations` schema (see [Database Schema](#database-schema))
- Mistral API key

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <repo-url>
cd a01-quranic-citation-transformer
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
DB_USER=...
DB_PASSWORD=...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=...
API_KEY=<your-mistral-api-key>
```

## Database Schema

The script expects a table `quranic_citations.citations` with:

| Column          | Type    | Notes                          |
|-----------------|---------|--------------------------------|
| `id`            | serial  | primary key                    |
| `subchapter_id` | text    |                                |
| `chapter_id`    | text    |                                |
| `tafsir_id`     | integer |                                |
| `pos`           | integer |                                |
| `text`          | text    | original (vocalized) citation  |
| `normalized_text` | text  | normalizedcitation             |
| `sura`          | integer | nullable, set by predict step  |
| `aya_start`     | integer | nullable, set by predict step  |
| `aya_end`       | integer | nullable, set by predict step  |

A unique constraint named `citations_subchapter_id_pos_key` on `(subchapter_id, pos)` is required for the upsert logic.

Here is the original SQL Create statement:

```sql
-- Table: quranic_citations.citations

-- DROP TABLE IF EXISTS quranic_citations.citations;

CREATE TABLE IF NOT EXISTS quranic_citations.citations
(
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1 ),
    subchapter_id text COLLATE pg_catalog."default" NOT NULL,
    text text COLLATE pg_catalog."default" NOT NULL,
    normalized_text text COLLATE pg_catalog."default",
    pos integer NOT NULL,
    sura integer,
    aya_start integer,
    aya_end integer,
    translation text COLLATE pg_catalog."default",
    chapter_id text COLLATE pg_catalog."default" NOT NULL,
    tafsir_id integer,
    CONSTRAINT citations_pkey PRIMARY KEY (id),
    CONSTRAINT citations_subchapter_id_pos_key UNIQUE (subchapter_id, pos),
    CONSTRAINT chapter_id FOREIGN KEY (chapter_id)
        REFERENCES tafsir.chapter (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE SET NULL,
    CONSTRAINT tafsir_id FOREIGN KEY (tafsir_id)
        REFERENCES metadata.tafsir (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE SET NULL
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS quranic_citations.citations
    OWNER to tommy;

```

## Usage

### Insert citations from input files

Place text files in the `input/` directory. The filename encodes the IDs:

- Filename pattern: `sc.<tafsir_id>_<chapter>_<rest>.txt`
- Example: `sc.1_37_22_23.txt` → `tafsir_id=1`, `chapter_id=c.1_37`, `subchapter_id=sc.1_37_22_23`

Each citation inside the file must be wrapped in curly braces:

```
Some surrounding text {الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ} more commentary
{الرَّحْمَٰنِ الرَّحِيمِ}
```

Run:

```bash
uv run python citations.py -i
```

### Predict sura/aya for citations

```bash
uv run python citations.py -p
```

The script sends each unresolved citation to the Mistral API and writes the predicted `sura`, `aya_start`, `aya_end` back to the database. Predictions are checkpointed every 25 calls, so a crash mid-run will not lose all progress.

Both flags can be combined to run insert and predict back-to-back:

```bash
uv run python citations.py -i -p
```

## How It Works

### Arabic text normalization

Citations are normalized internally for reliable comparison and deduplication:

- Unicode NFKC normalization
- Removal of Tatweel and diacritics (Tashkeel, Quranic annotation signs)
- Alef variants (أ, إ, آ, ٱ) → ا
- Teh Marbuta (ة) → ه
- Yeh variants (ى, ی) → ي
- Whitespace collapsed

The original (vocalized) text is sent to the LLM, since canonical Quranic verses are typically vocalized; the normalized form is used only as a stable join key for matching responses back to rows.

### Protected upsert

The `ON CONFLICT DO UPDATE` clause includes a `WHERE citations.sura IS NULL` condition, so already-predicted entries are never overwritten when re-running the insert step. To force a re-prediction, manually set `sura` back to `NULL` in the database and run `-p` again.

## License


