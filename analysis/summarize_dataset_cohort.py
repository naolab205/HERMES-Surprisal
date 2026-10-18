#!/usr/bin/env python3
"""Audit the manuscript cohort counts from archived predictions and epoch metadata."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = Path('/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms')
PARTICIPANTS = Path('/data/my/ds004952/participants.tsv')


def main() -> None:
    pred = pd.read_csv(ROOT / 'results/tables/charbag_MM_mid_predictions.csv')
    people = pd.read_csv(PARTICIPANTS, sep='\t').set_index('participant_id')
    rows = []
    for book, part in pred.groupby('book'):
        files = list(META.glob(f'meta_*_{book}.csv'))
        epochs = sum(sum(1 for _ in open(f, 'rb')) - 1 for f in files)
        demo = people.loc[part.subject.unique()]
        rows.append(dict(book=book, participants=part.subject.nunique(),
                         female=int(demo.sex.eq('F').sum()),
                         male=int(demo.sex.eq('M').sum()),
                         right_handed=int(demo.hand.eq('R').sum()),
                         subject_book_recordings=len(files),
                         row_readings=len(part), unique_rows=part.item_id.nunique(),
                         character_epochs=epochs))
    out = pd.DataFrame(rows)
    assert out.row_readings.sum() == len(pred) == 102499
    assert out.character_epochs.sum() == 917096
    out.to_csv(ROOT / 'analysis/dataset_cohort_summary.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
