"""Package only active manuscript dependencies and audit the current revision."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
TEX=ROOT/'els-cas-templates'
OUT=ROOT/'analysis/window_explanation_20261009'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=(TEX/'main.tex').read_text()
    abstract=source.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    word_count=len(abstract.split())
    assert word_count<=250
    pages=len(pymupdf.open(TEX/'main.pdf'))
    assert pages<=20
    figures=re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',source)
    assert len(figures)==7
    body=re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '',source,flags=re.S)
    body=body.split('\\section*{Declaration of generative AI')[0]
    labels=re.findall(r'\\label\{(fig:[^}]+)\}',source)
    references={f:body.count('\\ref{'+f+'}') for f in labels}
    assert all(n>=1 for n in references.values())
    citations=set()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}',source):citations.update(group.split(','))
    assert len(citations)>=15
    for key in citations:assert re.search(r'@\w+\{'+re.escape(key)+',',(TEX/'references.bib').read_text())
    with zipfile.ZipFile('/data/2_readEEG/els-cas-templates.zip') as template:
        matches={name:template.read('els-cas-templates/'+name)==(TEX/name).read_bytes()
            for name in ['cas-dc.cls','cas-common.sty','cas-model2-names.bst']}
    assert all(matches.values())
    files=[TEX/n for n in ['main.tex','references.bib','main.bbl','cas-dc.cls',
                           'cas-common.sty','cas-model2-names.bst']]
    files += [TEX/n for n in figures]
    files += sorted(p for p in (TEX/'thumbnails').rglob('*') if p.is_file())
    archive=ROOT/'KBS_Surprisal_Overleaf_CAS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(TEX))
        z.write(ROOT/'els-cas-templates/submission_notes/highlights.txt','highlights.txt')
        z.writestr('README.txt','Main document: main.tex. Elsevier CAS double-column template. Use pdfLaTeX and BibTeX in Overleaf. Scientific evidence remains retrospective development only. Individual CRediT roles await author input.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    hashed=files+[TEX/'main.pdf',ROOT/'els-cas-templates/submission_notes/highlights.txt',ROOT/'analysis/redraw_primary_fusion_20261008.py',ROOT/'analysis/restore_auc_and_history_20261008.py',ROOT/'model_code/src/foundation/hiernet.py',
        ROOT/'model_code/src/foundation/support_consensus.py',ROOT/'model_code/scripts/run_support_consensus_cv.py']
    hashed += [ROOT/'results/window_alignment_20261008'/n for n in ['PROTOCOL.json','summary.csv','complete_metrics.csv','paired_differences.json','eye_timestamp_audit.json']]
    hashed += [ROOT/'results/tables'/n for n in ['charbag_MM_mid_summary.json','charbag_MM_mid_per_fold.csv','charbag_MM_mid_predictions.csv']]
    run=ROOT/'results/support_consensus_20261006'
    hashed += [run/'complete_metrics.csv',run/'PROTOCOL.json',run/'paired_bootstrap.json']
    hashed += [run/'consensus_source/checkpoints'/f'consensus_source_f{f}.pt' for f in [0,3]]
    (OUT/'revision_checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in hashed},indent=2))
    report={'date':'2026-10-09','pages':pages,'abstract_words':word_count,
        'figures':len(figures),'tables':len(re.findall(r'\\begin\{table\*?\}',source)),
        'cited_references':len(citations),'figure_prose_references':references,
        'original_template_matches':matches,'archive_sha256':sha(archive),
        'main_pdf_sha256':sha(TEX/'main.pdf'),'archive_entries':len(files)+2,
        'source_age':'Published ChineseEEG: 18–29, mean22.7; local README18–24/20.68 differs',
        'evidence':'Primary archived MM_mid seven-block EEG–Eye results: item rho0.405±0.015, item AUC0.748±0.010; strict two-setting candidates remain secondary and unpromoted',
        'pending_metadata':['Individual factual CRediT roles'],
        'journal_guide':'Previous direct KBS guide access returned403; supplied<=250words/<=20pages checked',
        'ai_policy':'Current Elsevier explanatory-image policy checked; captions/general declaration disclose imagegen',
        'layout':'Rendered PDF inspected; no premature forced breaks. CAS frontmatter retains one empty-box overfull warning; no visible clipping.',
        'independent_archive_build':'Verify separately after packaging; see package_build_verification.json'}
    (OUT/'manuscript_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
