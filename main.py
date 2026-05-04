"""
Quranic citations: extract from input files and predict sura/aya via LLM.
"""

import argparse
import os
import re
import unicodedata
from typing import List, Tuple, TypedDict

import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

load_dotenv()


# ---------- Types ----------


class CitationDict(TypedDict):
    subchapter_id: str
    chapter_id: str
    tafsir_id: int
    pos: int
    text: str
    normalize_text: str


class LLMResponse(BaseModel):
    text: str
    sura: int
    aya_start: int
    aya_end: int


# ---------- Config ----------

DB = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
API_KEY = os.getenv("API_KEY")
MODEL = "mistral-large-latest"  # Quran identification needs a stronger model
INPUT_DIR = "input"
TABLE = "citations"
SCHEMA = "quranic_citations"


# ---------- Arabic normalization ----------


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for reliable string comparison (used internally only)."""
    if not isinstance(text, str):
        return text

    # Unify Unicode composition
    text = unicodedata.normalize("NFKC", text)
    # Remove Tatweel (decorative)
    text = text.replace("\u0640", "")
    # Remove diacritics (Tashkeel + Quranic annotation signs)
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]",
        "",
        text,
    )
    # Normalize Alef variants -> plain Alef
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Normalize Teh Marbuta -> Heh
    text = text.replace("ة", "ه")
    # Normalize Yeh variants -> plain Yeh
    text = re.sub(r"[ىی]", "ي", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- DB helpers ----------


def upsert(table, conn, keys, data_iter):
    """Upsert helper passed as `method` to DataFrame.to_sql.

    On conflict, only update rows where `sura` is still NULL. This protects
    already-predicted entries from being overwritten when re-running the insert
    step (e.g. if the same citation is parsed again from the input folder).
    Newly inserted rows are unaffected — the WHERE clause only restricts the
    UPDATE branch of ON CONFLICT.

    Note: we use `column("sura")` rather than `table.table.c.sura` because
    pandas constructs `table.table` from the DataFrame's columns only. In the
    insert step the DataFrame has no `sura` column, so attribute access would
    fail. An unqualified column reference inside ON CONFLICT DO UPDATE WHERE
    resolves to the existing target row, which is what we want.
    """
    stmt = insert(table.table).values(list(data_iter))
    stmt = stmt.on_conflict_do_update(
        constraint="citations_subchapter_id_pos_key",
        set_={k: stmt.excluded[k] for k in keys if k != "id"},
        where=text("citations.sura IS NULL"),
    )
    conn.execute(stmt)


def load_df() -> pd.DataFrame:
    """Load all citations and add an internal normalized_text column."""
    df = pd.read_sql_table(TABLE, DB, schema=SCHEMA, index_col="id")
    # We only load data that has not been predicted yet
    df = df[df["sura"].isna()]
    return df


def write_to_db(df: pd.DataFrame) -> None:
    """Persist a DataFrame to the citations table.

    The `normalized_text` column is for in-memory use only and must be dropped
    before writing, since it does not exist in the DB schema.
    """
    df = df.drop_duplicates(subset=["subchapter_id", "pos"], keep="last")
    df.to_sql(
        name=TABLE,
        schema=SCHEMA,
        con=DB,
        if_exists="append",
        index=False,
        method=upsert,
    )


def normalize_quran_db() -> None:
    """One-time function to normalize a Aya table in databse.

    This was necessary to enable the match-search."""
    df = pd.read_sql_table("aya", DB, schema="quran", index_col="index")
    df["normalized_text"] = df["content"].map(normalize_arabic)
    # Creating a new table to asure the old DB stays as is
    df.to_sql(
        name="aya_normalized",
        schema="quran",
        con=DB,
        if_exists="replace",
        index=True,
    )


# ---------- Insert step ----------


def parse_citations(text: str) -> List[str]:
    """Extract all {…} citations from a file's contents."""
    return [c.strip() for c in re.findall(r"\{(.*?)\}", text)]


def parse_filename(filename: str) -> Tuple[str, str, int]:
    """Derive (subchapter_id, chapter_id, tafsir_id) from a filename."""
    sc_id = os.path.splitext(filename)[0]
    tafsir_id = int(sc_id.split("_")[0].split(".")[1])
    chapter_id = f"c.{tafsir_id}_{sc_id.split('_')[1]}"
    return sc_id, chapter_id, tafsir_id


def collect_citations_from_files() -> List[CitationDict]:
    """Read all files in INPUT_DIR and return their citations as dict rows."""
    rows: List[CitationDict] = []
    for filename in os.listdir(INPUT_DIR):
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        sc_id, chapter_id, tafsir_id = parse_filename(filename)
        print(f"Processing {sc_id} (chapter={chapter_id}, tafsir={tafsir_id})")

        with open(filepath, "r", encoding="utf-8") as f:
            citations = parse_citations(f.read())

        for idx, text in enumerate(citations):
            rows.append(
                {
                    "subchapter_id": sc_id,
                    "chapter_id": chapter_id,
                    "tafsir_id": tafsir_id,
                    "pos": idx,
                    "text": text,
                    "normalized_text": normalize_arabic(text),
                }
            )
    return rows


def run_insert() -> None:
    """Read input/ files, parse citations, upsert into DB."""
    rows = collect_citations_from_files()
    if not rows:
        print("No citations found in input/.")
        return

    df_new = pd.DataFrame(rows)
    print(f"Inserting {len(df_new)} citations...")
    write_to_db(df_new)
    print("Insert done.")


# ---------- Predict step ----------


def predict_one(client: Mistral, original_text: str) -> LLMResponse | None:
    """Call the LLM for a single citation. Returns None on failure."""
    msg = [
        {
            "role": "user",
            "content": (
                "Predict the sura and ayas of the following verse from the Quran.\n\n"
                "######\n\n"
                f"{original_text}"
            ),
        }
    ]
    try:
        chat_response = client.chat.parse(
            model=MODEL, messages=msg, response_format=LLMResponse
        )
        return chat_response.choices[0].message.parsed
    except Exception as e:
        print(f"  ! API error: {e}")
        return None


def run_match(df: pd.DataFrame) -> None:
    """Match sura/aya in Quran."""
    print("Matching Sura/Ayat in Quran... this may take a while.")
    # Loading aya table from DB
    df_aya = pd.read_sql_table("aya_normalized", DB, "quran", index_col="index")
    # Matching normalized quotes with normalized ayat in Quran
    # Note that there is no multi-aya matching currently and aya_end_m / aya_start_m are always the same
    df["sura_m"] = df["normalized_text"].map(lambda x: df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)]["sura_id"].iloc[0] if not df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)].empty else None).astype("Int64")
    df["aya_start_m"] = df["normalized_text"].map(lambda x: df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)]["aya_id"].iloc[0] if not df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)].empty else None).astype("Int64")
    df["aya_end_m"] = df["normalized_text"].map(lambda x: df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)]["aya_id"].iloc[0] if not df_aya[df_aya["normalized_text"].str.contains(x, case=False, na=False, regex=False)].empty else None).astype("Int64")
    df.to_sql(
        name=TABLE,
        schema=SCHEMA,
        con=DB,
        if_exists="replace",
        index=True,
        index_label="id",
    )


def run_predict(df: pd.DataFrame) -> None:
    """Predict sura/aya for all rows lacking a prediction."""
    if not API_KEY:
        print("No API_KEY found in environment!")
        return

    # Only rows without a prediction; deduplicate to avoid redundant API calls.
    pending = df[df["sura"].isna()].drop_duplicates("normalized_text")
    print(f"Total rows: {len(df)} | Rows to predict: {len(pending)}")

    if pending.empty:
        return

    client = Mistral(api_key=API_KEY)
    # Send the ORIGINAL text to the LLM (vocalized form is what the model knows),
    # but keep the normalized form as a stable join key.
    pairs = list(zip(pending["normalized_text"], pending["text"]))

    predictions: List[Tuple[str, LLMResponse]] = []
    for idx, (norm_text, original_text) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] requesting...")
        result = predict_one(client, original_text)
        if result is not None:
            # Pair with the normalized key we sent for, NOT result.text — the
            # model often returns a canonical (vocalized) form that wouldn't match.
            predictions.append((norm_text, result))

        # Periodic checkpoint so a late crash doesn't lose everything.
        if idx % 25 == 0:
            apply_and_save(df, predictions)
            predictions = []

    if predictions:
        apply_and_save(df, predictions)

    print("Predict done.")


def show_stats() -> None:
    """Display differences between LLM predicted sura/aya and direkt sura/aya matching in Quran."""
    df = pd.read_sql_table("citations_debug", DB, schema=SCHEMA, index_col="id")
    df_diff = df[(df["sura"] != df["sura_m"]) | (df["aya_start"] != df["aya_start_m"])]
    print(f"There are {df_diff.shape[0]} mismatches between LLM prediction and the direct matching approach. See the diff file for details.")
    df_diff.to_csv("diff.csv")


def apply_and_save(
    df: pd.DataFrame, predictions: List[Tuple[str, LLMResponse]]
) -> None:
    """Write a batch of predictions back into df and persist to DB."""
    if not predictions:
        return

    for norm_key, parsed in predictions:
        df.loc[
            df["normalized_text"] == norm_key,
            ["sura", "aya_start", "aya_end"],
        ] = (parsed.sura, parsed.aya_start, parsed.aya_end)

    write_to_db(df)
    print(f"  saved batch of {len(predictions)} predictions")


# ---------- Entry point ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Quranic citations and predict sura/aya."
    )
    parser.add_argument(
        "-i",
        "--insert",
        action="store_true",
        help=f"Read citations from {INPUT_DIR}/ and upsert into DB.",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Predict sura/aya for unresolved citations.",
    )
    parser.add_argument(
        "-m",
        "--match",
        action="store_true",
        help="Predict sura/aya based on direct matching with Quran. Does currently not work for multi-aya quotes.",
    )
    parser.add_argument(
            "-nt",
            "--normalize-table",
            action="store_true",
            help="One-time function to normalize aya table in DB. Should only be called once and depends on the database (should most likely never be used outside A01 project)."
            )
    parser.add_argument(
            "-s",
            "--stats",
            action="store_true",
            help="Show differences between LLM predictions and parsing of sura/aya."
            )
    args = parser.parse_args()

    if not (args.insert or args.predict or args.normalize_table or args.match or args.stats):
        parser.print_help()
        return

    if args.insert:
        run_insert()

    if args.predict:
        df = load_df()
        if df.empty:
            print("No data loaded from DB.")
            return
        run_predict(df)

    if args.match:
        # We will always rematch entire table
        df = pd.read_sql_table(TABLE, DB, schema=SCHEMA, index_col="id")
        if df.empty:
            print("No data loaded from DB.")
            return
        run_match(df)

    if args.stats:
        show_stats()

    if args.normalize_table:
        normalize_quran_db()


if __name__ == "__main__":
    main()
