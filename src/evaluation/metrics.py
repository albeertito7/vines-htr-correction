"""
Evaluation metrics for HTR post-correction quality.

Metrics:
    - CER (Character Error Rate): edit distance at character level divided
      by the ground truth length. The primary metric for HTR evaluation.
    - WER (Word Error Rate): same as CER but at word level.
    - Overcorrection: words that were correct in the HTR output but were
      incorrectly changed by the LLM.
    - Precision/Recall/F1: of the corrections made by the LLM, how many
      were correct (precision), how many real errors were caught (recall).

All page-level metrics are computed on concatenated full text (not line
by line) to avoid segmentation alignment issues between HTR and GT.
"""


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein edit distance between two sequences.
    Works with both strings (for CER) and word lists (for WER).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(ground_truth, hypothesis):
    """
    Calculate Character Error Rate.

    Args:
        ground_truth: reference correct text.
        hypothesis: text to evaluate (HTR raw or corrected).

    Returns:
        float between 0 and 1 (0 = perfect, 1 = all wrong).
        Can exceed 1.0 if the hypothesis is much longer than the reference.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = levenshtein_distance(ground_truth, hypothesis)
    return distance / len(ground_truth)


def calculate_wer(ground_truth, hypothesis):
    """
    Calculate Word Error Rate.

    Args:
        ground_truth: reference correct text.
        hypothesis: text to evaluate.

    Returns:
        float between 0 and 1 (can exceed 1.0 if hypothesis is longer).
    """
    gt_words = ground_truth.split()
    hyp_words = hypothesis.split()

    if len(gt_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    distance = levenshtein_distance(gt_words, hyp_words)
    return distance / len(gt_words)


def calculate_overcorrections(ground_truth, htr_text, corrected_text):
    """
    Count overcorrections: words that were correct in the HTR output
    but were incorrectly changed by the LLM.

    Uses positional word alignment (assumes similar word count across
    all three versions). This is a simplification that works well for
    full-page text where the LLM preserves the overall structure.

    Args:
        ground_truth: reference correct text.
        htr_text: original HTR text (before correction).
        corrected_text: text after LLM correction.

    Returns:
        dict with {count, words: [{original, corrected, expected}]}
    """
    gt_words = ground_truth.split()
    htr_words = htr_text.split()
    cor_words = corrected_text.split()

    overcorrections = []
    min_len = min(len(gt_words), len(htr_words), len(cor_words))

    for i in range(min_len):
        htr_correct = htr_words[i] == gt_words[i]
        llm_changed = htr_words[i] != cor_words[i]
        llm_wrong = cor_words[i] != gt_words[i]

        # HTR was correct, LLM changed it, and the change is wrong
        if htr_correct and llm_changed and llm_wrong:
            overcorrections.append({
                "original": htr_words[i],
                "corrected": cor_words[i],
                "expected": gt_words[i],
            })

    return {
        "count": len(overcorrections),
        "words": overcorrections,
    }


def calculate_corrections(ground_truth, htr_text, corrected_text):
    """
    Analyse corrections made by the LLM: how many are correct,
    how many are wrong, and how many errors were missed.

    Args:
        ground_truth: reference correct text.
        htr_text: original HTR text.
        corrected_text: LLM-corrected text.

    Returns:
        dict with {true_positives, false_positives, false_negatives,
                   precision, recall, f1}
    """
    gt_words = ground_truth.split()
    htr_words = htr_text.split()
    cor_words = corrected_text.split()

    true_positives = 0   # HTR errors correctly fixed by LLM
    false_positives = 0  # LLM changes that worsen or are unnecessary
    false_negatives = 0  # HTR errors that the LLM missed

    min_len = min(len(gt_words), len(htr_words), len(cor_words))

    for i in range(min_len):
        htr_wrong = htr_words[i] != gt_words[i]
        llm_changed = htr_words[i] != cor_words[i]
        llm_correct = cor_words[i] == gt_words[i]

        if htr_wrong and llm_changed and llm_correct:
            true_positives += 1
        elif llm_changed and not llm_correct:
            false_positives += 1
        elif htr_wrong and not llm_changed:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_lines(ground_truth_lines, htr_lines, corrected_lines):
    """
    Evaluate a page by comparing GT, HTR and corrected text line by line.

    Note: this function concatenates all lines into full text before
    computing metrics, so it is equivalent to evaluate_page(). It exists
    for API compatibility with callers that pass lists of lines.

    Args:
        ground_truth_lines: list of strings (correct text per line).
        htr_lines: list of strings (HTR text per line).
        corrected_lines: list of strings (corrected text per line).

    Returns:
        dict with global and per-correction metrics.
    """
    gt_full = " ".join(ground_truth_lines)
    htr_full = " ".join(htr_lines)
    cor_full = " ".join(corrected_lines)

    return {
        "cer_before": round(calculate_cer(gt_full, htr_full), 4),
        "cer_after": round(calculate_cer(gt_full, cor_full), 4),
        "wer_before": round(calculate_wer(gt_full, htr_full), 4),
        "wer_after": round(calculate_wer(gt_full, cor_full), 4),
        "overcorrections": calculate_overcorrections(gt_full, htr_full, cor_full),
        "corrections": calculate_corrections(gt_full, htr_full, cor_full),
    }

def evaluate_page(gt_lines, htr_lines, corrected_lines):
    """
    Evaluate a page by comparing full concatenated text.

    This is the primary evaluation function. It compares full-page text
    (all lines concatenated) rather than line by line, avoiding the
    segmentation mismatch problem between HTR and GT (which can differ
    by 1-11 lines on the same page).

    Args:
        gt_text: full ground truth text (string or list of line strings).
        htr_text: full HTR text (string or list of line strings).
        corrected_text: full corrected text (string or list of line strings).

    Returns:
        dict with CER/WER before and after, overcorrections, and corrections.
    """
    gt_full = " ".join(gt_lines)
    htr_full = " ".join(htr_lines)
    cor_full = " ".join(corrected_lines)

    return {
        "cer_before": round(calculate_cer(gt_full, htr_full), 4),
        "cer_after": round(calculate_cer(gt_full, cor_full), 4),
        "wer_before": round(calculate_wer(gt_full, htr_full), 4),
        "wer_after": round(calculate_wer(gt_full, cor_full), 4),
        "overcorrections": calculate_overcorrections(gt_full, htr_full, cor_full),
        "corrections": calculate_corrections(gt_full, htr_full, cor_full),
    }


if __name__ == "__main__":
    # Quick verification example
    gt = "Por la mañana fui a tomar mi baño en las Thermes de Bagnères"
    htr = "Por la mañana fui á tomar mi bano en las Thermes de Bagneres"
    cor = "Por la mañana fui a tomar mi baño en las Thermos de Bagnères"

    print("Ground truth:", gt)
    print("HTR:         ", htr)
    print("Corrected:    ", cor)
    print()

    print(f"CER before:  {calculate_cer(gt, htr):.4f}")
    print(f"CER after: {calculate_cer(gt, cor):.4f}")
    print(f"WER before:  {calculate_wer(gt, htr):.4f}")
    print(f"WER after: {calculate_wer(gt, cor):.4f}")
    print()

    overcor = calculate_overcorrections(gt, htr, cor)
    print(f"Overcorrections: {overcor['count']}")
    for w in overcor["words"]:
        print(f"  '{w['original']}' -> '{w['corrected']}' (expected '{w['expected']}')")
    print()

    corrections = calculate_corrections(gt, htr, cor)
    print(f"True positives: {corrections['true_positives']}")
    print(f"False positives:   {corrections['false_positives']}")
    print(f"False negatives:    {corrections['false_negatives']}")
    print(f"Precision: {corrections['precision']}")
    print(f"Recall:   {corrections['recall']}")
    print(f"F1:       {corrections['f1']}")
    print()

    # Test with evaluate_page (full text)
    result = evaluate_page(gt, htr, cor)
    print("evaluate_page:", result)
