import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Clean and normalize news article text.
    Preserves financial terms, currencies, percentages, and geopolitical entities.
    Removes HTML tags and wire-service boilerplate.
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove syndicated boilerplate prefixes (e.g., "LONDON (Reuters) -", "TORONTO, Oct 05 (GLOBE NEWSWIRE) --")
    text = re.sub(r"^[A-Z\s,]+(?:--|\(Reuters\)|\(Bloomberg\)|\(RTTNews\)|\(GLOBE NEWSWIRE\)|-|\:)\s*", "", text)
    text = re.sub(r"^entitle:\s*", "", text, flags=re.IGNORECASE)
    
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def compute_text_hash(text: str) -> str:
    """Generate SHA-256 hash of normalized text for exact duplicate detection."""
    norm = re.sub(r"[^\w\s]", "", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def compute_prefix_signature(text: str, num_words: int = 15) -> str:
    """Extract lead-word signature for syndicated news story matching."""
    words = re.findall(r"\b\w+\b", text.lower())
    lead = " ".join(words[:num_words])
    return hashlib.sha256(lead.encode("utf-8")).hexdigest()

def deduplicate_corpus(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: Optional[str] = "title",
    similarity_threshold: float = 0.85
) -> pd.DataFrame:
    """
    Remove exact and near-duplicate syndicated articles in O(N) time.
    Combines:
      1. Exact content hash deduplication
      2. Exact title deduplication
      3. Lead-sentence syndication signature hashing (near-duplicate wire filtering)
    Preserves earliest publication timestamp and source URL.
    """
    initial_count = len(df)
    df = df.copy()
    
    # Step 1: Clean text and compute full content hash
    df["_clean_text"] = df[text_col].astype(str).apply(clean_text)
    df = df[df["_clean_text"].str.len() > 15].copy()
    df["_content_hash"] = df["_clean_text"].apply(compute_text_hash)
    
    # Drop exact text duplicates
    df = df.drop_duplicates(subset=["_content_hash"], keep="first")
    
    # Step 2: Title deduplication if title exists
    if title_col and title_col in df.columns:
        df["_title_clean"] = df[title_col].astype(str).str.lower().str.strip()
        df = df[~(df["_title_clean"].str.len() > 10) | ~df.duplicated(subset=["_title_clean"], keep="first")]
        df = df.drop(columns=["_title_clean"])
        
    # Step 3: Near-duplicate wire syndication removal via lead-sentence signature
    df["_prefix_sig"] = df["_clean_text"].apply(compute_prefix_signature)
    df = df.drop_duplicates(subset=["_prefix_sig"], keep="first")
    
    # Clean temporary columns
    df = df.drop(columns=["_clean_text", "_content_hash", "_prefix_sig"], errors="ignore")
    
    removed = initial_count - len(df)
    print(f"[Deduplication] Removed {removed} duplicates ({initial_count} -> {len(df)} articles retained).")
    return df

def validate_schema_and_integrity(
    df: pd.DataFrame,
    required_cols: List[str] = ["text", "date", "url", "source", "topic"],
    is_labeled: bool = False
) -> Dict[str, Any]:
    """
    Strict validation of dataset schema, missing values, date coverage, and label distribution.
    Enforces the 'No synthetic data, No model predictions' rule.
    """
    results = {"valid": True, "errors": [], "warnings": [], "stats": {}}
    
    # 1. Check required columns
    for col in required_cols:
        if col not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Missing required column: {col}")
            
    if is_labeled and "label" not in df.columns:
        results["valid"] = False
        results["errors"].append("Missing 'label' column for human-labeled dataset.")
        
    if not results["valid"]:
        return results
        
    # 2. Check missing values
    null_counts = df[required_cols].isna().sum().to_dict()
    results["stats"]["null_counts"] = null_counts
    if null_counts.get("text", 0) > 0:
        results["valid"] = False
        results["errors"].append(f"Found {null_counts['text']} rows with null text.")
        
    # 3. Date validity & coverage
    try:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        null_dates = parsed_dates.isna().sum()
        if null_dates > 0:
            results["warnings"].append(f"Found {null_dates} rows with unparseable dates.")
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            results["stats"]["date_range"] = {
                "min": str(min_date),
                "max": str(max_date),
                "valid_count": int(len(valid_dates))
            }
            # Historical boundary check (2000-01-01 to 2026-11-11)
            if min_date < pd.Timestamp("2000-01-01"):
                results["warnings"].append(f"Articles precede 2000-01-01: {min_date}")
            if max_date > pd.Timestamp("2026-11-11 23:59:59"):
                results["errors"].append(f"Articles exceed historical ceiling 2026-11-11: {max_date}")
                results["valid"] = False
    except Exception as e:
        results["errors"].append(f"Date validation failed: {str(e)}")
        results["valid"] = False
        
    # 4. Check for synthetic / AI-generated artifacts
    banned_synthetic_keywords = ["glm-5", "gpt-4", "chatgpt", "synthetic", "ai-generated", "llama-"]
    text_sample = df["text"].astype(str).str.lower()
    for kw in banned_synthetic_keywords:
        match_count = text_sample.str.contains(kw).sum()
        if match_count > 0:
            results["warnings"].append(f"Found keyword '{kw}' in {match_count} articles. Verify data provenance.")
            
    # 5. Label validation if labeled
    if is_labeled and "label" in df.columns:
        label_dist = df["label"].value_counts().to_dict()
        results["stats"]["label_distribution"] = label_dist
        valid_label_set = {0, 1, 2, "positive", "negative", "neutral"}
        unique_labels = set(df["label"].unique())
        if not unique_labels.issubset(valid_label_set):
            results["valid"] = False
            results["errors"].append(f"Invalid labels found: {unique_labels - valid_label_set}")
            
    results["stats"]["total_rows"] = len(df)
    return results

def create_chronological_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly chronologically to prevent temporal lookahead leakage.
    Ensures:
      train_end <= val_start <= val_end <= test_start
    Also enforces zero duplicate leakage across partitions.
    """
    df = df.copy()
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Sort strictly by date ascending
    df = df.sort_values(by="_parsed_date", ascending=True).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    # Duplicate leakage check across partitions
    train_hashes = set(train_df["text"].apply(compute_text_hash))
    val_df = val_df[~val_df["text"].apply(compute_text_hash).isin(train_hashes)].reset_index(drop=True)
    
    train_val_hashes = train_hashes | set(val_df["text"].apply(compute_text_hash))
    test_df = test_df[~test_df["text"].apply(compute_text_hash).isin(train_val_hashes)].reset_index(drop=True)
    
    # Clean temporary column
    train_df = train_df.drop(columns=["_parsed_date"], errors="ignore")
    val_df = val_df.drop(columns=["_parsed_date"], errors="ignore")
    test_df = test_df.drop(columns=["_parsed_date"], errors="ignore")
    
    print(f"[Chronological Split] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df
