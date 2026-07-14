"""Self-learning tutorial: clinical NLP on real MIMIC-III discharge summaries.

This script is designed to be read top-to-bottom. It performs the same core
workflow shown in the accompanying presentation: build a disease-focused
cohort, extract entities with three NLP pipelines, train Word2Vec models, tune
t-SNE, and save only aggregate figures and tables. Do not publish MIMIC note
text or use this code outside the access terms that govern your MIMIC account.
"""

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import scispacy
import medspacy
from gensim.models import Word2Vec
from medspacy.ner import TargetRule
from scispacy.abbreviation import AbbreviationDetector
from sklearn.manifold import TSNE, trustworthiness


# Step 0: make the tutorial deterministic where possible.
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# Keep paths relative so a peer can rebuild the project after configuring the
# two locations below. The MIMIC directory must contain the full, authorized
# MIMIC-III database - never the demo database or copied note excerpts.
PROJECT_ROOT = Path(__file__).resolve().parent
MIMIC_ROOT = PROJECT_ROOT.parent / "mimic-iii-clinical-database-1.4" / "mimic-iii-clinical-database-1.4"
OUTPUT_DIR = PROJECT_ROOT / "self_learning_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_sah_discharge_summaries(limit=80):
    """Return de-duplicated discharge summaries for ICD-9 code 430.

    We first find admissions with subarachnoid hemorrhage in DIAGNOSES_ICD,
    then stream NOTEEVENTS in chunks. Chunking avoids loading the entire
    multi-gigabyte notes table into memory at once.
    """
    diagnoses_file = MIMIC_ROOT / "DIAGNOSES_ICD.csv.gz"
    notes_file = MIMIC_ROOT / "NOTEEVENTS.csv.gz"
    if not diagnoses_file.exists() or not notes_file.exists():
        raise FileNotFoundError("Configure MIMIC_ROOT to the authorized full MIMIC-III tables.")

    diagnoses = pd.read_csv(
        diagnoses_file,
        compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        dtype={"ICD9_CODE": str},
    )
    # Remove formatting punctuation before matching the exact ICD-9 diagnosis.
    sah_admissions = set(
        diagnoses.loc[
            diagnoses.ICD9_CODE.str.replace(".", "", regex=False).str.strip().eq("430"),
            "HADM_ID",
        ].dropna().astype(int)
    )

    selected = []
    for notes_chunk in pd.read_csv(
        notes_file,
        compression="gzip",
        chunksize=50_000,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
    ):
        keep = notes_chunk[
            notes_chunk.HADM_ID.isin(sah_admissions)
            & notes_chunk.CATEGORY.eq("Discharge summary")
        ].dropna(subset=["TEXT"])
        if not keep.empty:
            selected.append(keep)

    cohort = pd.concat(selected, ignore_index=True).drop_duplicates("TEXT").head(limit).copy()
    cohort["note_id"] = range(1, len(cohort) + 1)
    if len(cohort) < 30:
        raise ValueError("The tutorial needs at least 30 ICD-9 430 discharge summaries.")
    return cohort


# Step 1: define a transparent, editable vocabulary for the tutorial.
# A controlled vocabulary keeps the comparison reproducible. In a production
# project, replace or supplement it with a validated pretrained NER model and
# an annotated evaluation set.
TERMS = {
    "CONDITION": [
        "subarachnoid hemorrhage", "SAH", "aneurysm", "headache", "stroke", "seizure",
        "hydrocephalus", "hypertension", "diabetes", "pneumonia", "vasospasm", "meningitis", "infarct",
    ],
    "DRUG": [
        "nimodipine", "levetiracetam", "Keppra", "aspirin", "heparin", "warfarin", "coumadin",
        "labetalol", "Tylenol", "acetaminophen", "morphine", "dexamethasone",
    ],
    "PROCEDURE": [
        "craniotomy", "angiography", "embolization", "coiling", "clipping", "intubation",
        "ventriculostomy", "CT scan", "MRI", "lumbar puncture",
    ],
    "ANATOMY": ["brain", "cerebral artery", "middle cerebral artery", "carotid artery", "ventricle", "neck", "head"],
}


def build_pipelines():
    """Build spaCy, scispaCy, and medSpaCy extractors with matched terms."""
    # spaCy: a clear general NLP baseline with broad clinical categories.
    spacy_nlp = spacy.blank("en")
    spacy_ruler = spacy_nlp.add_pipe("entity_ruler")
    spacy_ruler.add_patterns([{"label": label, "pattern": term} for label, values in TERMS.items() for term in values])

    # scispaCy: add biomedical abbreviation handling, then apply the same
    # vocabulary under biomedical labels. The matched vocabulary makes this a
    # controlled tutorial comparison rather than a claim of benchmark accuracy.
    sci_nlp = spacy.blank("en")
    sci_nlp.add_pipe("sentencizer")
    sci_nlp.add_pipe("abbreviation_detector")
    sci_ruler = sci_nlp.add_pipe("entity_ruler")
    sci_ruler.add_patterns([
        {"label": "DISEASE" if label == "CONDITION" else "CHEMICAL" if label == "DRUG" else label, "pattern": term}
        for label, values in TERMS.items() for term in values
    ])

    # medSpaCy: TargetMatcher identifies concepts and ConText attaches clinical
    # context attributes such as negation, historical status, and hypothetical status.
    med_nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
    target_matcher = med_nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add([
        TargetRule(term, "PROBLEM" if label == "CONDITION" else "MEDICATION" if label == "DRUG" else label)
        for label, values in TERMS.items() for term in values
    ])
    return {"spaCy": spacy_nlp, "scispaCy": sci_nlp, "medSpaCy": med_nlp}


def unique_entities(doc):
    """Avoid duplicate spans while preserving the original document order."""
    seen, results = set(), []
    for entity in doc.ents:
        key = (entity.start_char, entity.end_char, entity.label_)
        if key not in seen:
            seen.add(key)
            results.append(entity)
    return results


def extract_entities(cohort, pipelines):
    """Create one entity sequence per note for each library and an audit table."""
    records = []
    corpora = {name: [] for name in pipelines}
    for row in cohort.itertuples():
        for library, nlp in pipelines.items():
            tokens = []
            for entity in unique_entities(nlp(row.TEXT)):
                normalized = re.sub(r"[^a-z0-9]+", "_", entity.text.lower()).strip("_")
                if normalized:
                    tokens.append(normalized)
                records.append({
                    "library": library,
                    "note_id": row.note_id,
                    "entity": entity.text,
                    "normalized": normalized,
                    "label": entity.label_,
                    "negated": bool(getattr(entity._, "is_negated", False)) if library == "medSpaCy" else False,
                })
            corpora[library].append(tokens)
    return pd.DataFrame(records), corpora


def train_word2vec(corpora):
    """Train a small, reproducible entity Word2Vec model for each pipeline."""
    models = {}
    for library, sentences in corpora.items():
        usable_sentences = [sentence for sentence in sentences if sentence]
        models[library] = Word2Vec(
            usable_sentences,
            vector_size=50,
            window=5,
            min_count=1,
            sg=1,
            epochs=150,
            workers=1,
            seed=SEED,
        )
    return models


def tune_and_plot_tsne(models, corpora):
    """Select t-SNE perplexity by local-neighborhood trustworthiness."""
    best, tuning_rows = {}, []
    for library, model in models.items():
        words = np.array(model.wv.index_to_key)
        vectors = model.wv[words]
        for perplexity in [value for value in [5, 10, 20, 30] if value < len(words)]:
            coordinates = TSNE(
                n_components=2, perplexity=perplexity, learning_rate="auto",
                init="pca", max_iter=2000, random_state=SEED,
            ).fit_transform(vectors)
            score = trustworthiness(vectors, coordinates, n_neighbors=min(5, len(words) - 1))
            tuning_rows.append({"library": library, "perplexity": perplexity, "trustworthiness": score})
            if library not in best or score > best[library][0]:
                best[library] = (score, perplexity, coordinates, words)

    tuning = pd.DataFrame(tuning_rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"spaCy": "#0B6E99", "scispaCy": "#7A5195", "medSpaCy": "#D95F02"}
    for axis, (library, (score, perplexity, coordinates, words)) in zip(axes, best.items()):
        axis.scatter(coordinates[:, 0], coordinates[:, 1], color=colors[library], s=40, alpha=0.8)
        axis.set_title(f"{library}: perplexity={perplexity}, trustworthiness={score:.3f}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Tuned t-SNE of entity Word2Vec embeddings")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsne_all_libraries.png", dpi=180, bbox_inches="tight")
    return tuning, best


if __name__ == "__main__":
    cohort = load_sah_discharge_summaries()
    pipelines = build_pipelines()
    entities, corpora = extract_entities(cohort, pipelines)
    models = train_word2vec(corpora)
    tuning, best_tsne = tune_and_plot_tsne(models, corpora)

    # Save only aggregate tables; do not export raw MIMIC note text.
    summary = entities.groupby("library").agg(
        mentions=("entity", "size"),
        unique_entities=("normalized", "nunique"),
        notes_with_entities=("note_id", "nunique"),
    )
    summary.to_csv(OUTPUT_DIR / "entity_summary.csv")
    tuning.to_csv(OUTPUT_DIR / "tsne_tuning.csv", index=False)
    print(summary)
    print("Saved aggregate outputs to:", OUTPUT_DIR)
