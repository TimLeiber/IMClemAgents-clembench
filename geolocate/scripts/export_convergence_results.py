"""Export the saved replay comparison as CSV and a browsable HTML table.

This only formats existing scores; it never calls a model or changes results.
"""
import argparse
import csv
import html
import json
import math
import hashlib
from pathlib import Path


def metric_summary(predictions):
    played = [p for p in predictions if p is not None]
    n = len(predictions)
    def mean(key):
        return sum(p[key] for p in played) / len(played) if played else None
    def score(key):
        return sum(p[key] for p in played) / n if n else None
    return {'Episodes': n, '% Played': 100 * len(played) / n,
            'Inverse quality': mean('inverse'), 'Inverse clemscore': score('inverse'),
            'Exp 100 km quality': mean('exp100'), 'Exp 100 km adjusted score': score('exp100'),
            'Exp 50 km quality': mean('exp50'), 'Exp 50 km adjusted score': score('exp50'),
            'Exp 5 km quality': mean('exp5'), 'Exp 5 km adjusted score': score('exp5'),
            'Country correct % (played)': 100 * mean('country') if played else None,
            'Mean distance km (played)': mean('distance')}


def prediction_metrics(prediction, replay=False):
    distance = prediction['distance_km']
    return {'inverse': prediction['inverse_distance_quality' if replay else 'inverse_quality'],
            'exp100': prediction['exponential_distance_quality' if replay else 'exponential_quality'],
            'exp50': 100 * 2 ** (-distance / 50),
            'exp5': prediction['close_range_quality'],
            'country': float(prediction['country_correct']), 'distance': distance}


def export_results(run_dir: Path) -> None:
    summary = json.loads((run_dir / 'summary.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    source_dir = Path(manifest['configuration']['results_dir'])
    config = manifest['configuration']['scoring_config']
    assert config['exponential_distance_half_score_km'] == 100
    assert config['close_range_half_score_km'] == 5
    replays = {r['source_episode']: r for r in summary['episodes']}
    details, vanilla, model_instances = [], {}, {}
    for agent, result in summary['by_agent'].items():
        model = agent.split('-with-', 1)[1]
        cohort = [r for r in manifest['cohort'] if r['agent'] == agent]
        identities = {(r['experiment'], r['game_id']) for r in cohort}
        if model in model_instances and model_instances[model] != identities:
            raise ValueError(f'Mismatched instance sets for {model}')
        model_instances[model] = identities
        before, after = [], []
        for row in cohort:
            path = source_dir / agent / 'geolocate' / row['experiment'] / f"instance_{row['game_id']:05d}" / 'interactions.json'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['source_interactions_sha256']:
                raise ValueError(f'Source changed since replay: {path}')
            interaction = json.loads(raw)
            predictions = interaction.get('episode_result', {}).get('predictions') or []
            original = prediction_metrics(predictions[-1]) if predictions and not interaction.get('Aborted') else None
            before.append(original)
            replay = replays.get(row['source_episode'])
            after.append(prediction_metrics(replay['score'], replay=True)
                         if replay and replay.get('valid_response') else original)
        for condition, values, expected in [('Original', before, result['baseline']),
                                             ('Recovered', after, result['counterfactual'])]:
            if expected is None or result['pending_replays']:
                raise ValueError(f'Replay is incomplete for {agent}')
            metrics = metric_summary(values)
            if not math.isclose(metrics['Inverse clemscore'], expected['clemscore'], abs_tol=1e-9):
                raise ValueError(f'Score mismatch for {agent}')
            details.append({'Configuration': agent, 'Condition': condition, **metrics})
    for model, identities in sorted(model_instances.items()):
        values = []
        for experiment, game_id in sorted(identities):
            path = source_dir / model / 'geolocate' / experiment / f'instance_{game_id:05d}' / 'interactions.json'
            interaction = json.loads(path.read_text())
            predictions = interaction.get('episode_result', {}).get('predictions') or []
            values.append(prediction_metrics(predictions[-1]) if predictions and not interaction.get('Aborted') else None)
        vanilla[model] = metric_summary(values)
        details.append({'Configuration': model, 'Condition': 'Vanilla', **vanilla[model]})
    columns = ['Configuration', 'Episodes', 'Original clemscore',
               'Recovered clemscore', 'Delta', 'Vanilla clemscore', 'Recovered minus vanilla', 'Original % Played',
               'Recovered % Played', 'Original Quality', 'Recovered Quality',
               'Recovered timeouts', 'Selected timeouts']
    rows = []
    for agent, result in summary['by_agent'].items():
        before, after = result['baseline'], result['counterfactual']
        if after is None or result['pending_replays']:
            raise ValueError(f'Replay is incomplete for {agent}')
        if before['episodes'] != after['episodes']:
            raise ValueError(f'Comparison denominators differ for {agent}')
        reference = vanilla[agent.split('-with-', 1)[1]]['Inverse clemscore']
        rows.append([agent, before['episodes'], before['clemscore'],
                     after['clemscore'], after['clemscore'] - before['clemscore'],
                     reference, after['clemscore'] - reference,
                     before['percent_played'], after['percent_played'],
                     before['quality'], after['quality'],
                     result['rescued_timeouts'], result['selected_timeouts']])
    rows.sort(key=lambda row: row[3], reverse=True)
    with (run_dir / 'results.csv').open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
    with (run_dir / 'metrics.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    headers = ''.join(f'<th scope="col">{html.escape(name)}</th>' for name in columns)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            text = '—' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
            if index == 0:
                cells.append(f'<th scope="row">{html.escape(text)}</th>')
            else:
                cells.append(f'<td>{html.escape(text)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geolocate counterfactual results</title>
<style>
body {font: 15px system-ui, sans-serif; margin: 28px; color: #17202a; background: white}
h1 {font-size: 24px} p {max-width: 1050px; line-height: 1.5}
.table-wrap {overflow-x: auto} table {border-collapse: collapse; width: 100%}
th, td {padding: 12px 10px; border: 1px solid #d7dee5; text-align: right}
thead th {background: #eaf0f5; vertical-align: bottom; min-width: 85px}
tbody th {text-align: left; white-space: nowrap; font-weight: 500}
tbody tr:nth-child(even) {background: #f7f9fb}
td:nth-child(4), td:nth-child(5) {font-weight: 650}
</style></head><body><h1>Geolocate counterfactual results</h1>
<p>Original harness performance compared with performance after replacing selected timeout
outcomes with one-shot counterfactual predictions. Both columns include the same full set
of episodes per configuration. All non-selected outcomes remain unchanged.</p>
<p>Quality uses the inverse-distance metric. Clemscore is quality multiplied by the played
fraction. Recovery is additional inference, not a completed original harness run.
Sorted by recovered clemscore, descending. <a href="results.csv">Download CSV</a></p>
<div class="table-wrap"><table><thead><tr>'''
    document += headers + '</tr></thead><tbody>' + ''.join(body)
    document += '</tbody></table></div><h2>All metrics, including vanilla</h2>'
    document += '<p>Each row covers the same 15 instances. Quality, country correctness and mean distance are conditional on played episodes. '
    document += 'Adjusted scores count aborted episodes as zero, like clemscore. Exponential labels specify the half-score distance: '
    document += '100 × 2<sup>−distance / half-score distance</sup>. The 50 km metric is calculated from saved distances; '
    document += '100 km and 5 km are the logged metrics. Lower distance is better; higher scores are better. '
    document += '<a href="metrics.csv">Download all metrics</a>.</p>'
    document += '<p>Recovery uses additional inference with GLM reasoning low and Qwen reasoning off. '
    document += 'Vanilla and original harness runs retain their recorded configurations; this is not an equal-compute comparison.</p>'
    document += '<div class="table-wrap"><table><thead><tr>'
    document += ''.join(f'<th scope="col">{html.escape(c)}</th>' for c in details[0])
    document += '</tr></thead><tbody>'
    for row in sorted(details, key=lambda r: (r['Configuration'], r['Condition'])):
        document += '<tr>'
        for i, value in enumerate(row.values()):
            text = '—' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
            tag = 'th scope="row"' if i == 0 else 'td'
            document += f'<{tag}>{html.escape(text)}</{tag.split()[0]}>'
        document += '</tr>'
    document += '</tbody></table></div><p>Sources: saved replay summary, manifest, and matched official interactions. Values displayed to two decimal places.</p></body></html>\n'
    if summary['configuration'].get('guess_mode') == 'remaining-guesses':
        document = document.replace('one-shot counterfactual predictions',
                                    'counterfactual predictions using the remaining original guess allowance and normal feedback')
    (run_dir / 'results.html').write_text(document)
    print(f'Exported {len(rows)} configurations to {run_dir / "results.html"} and results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    export_results(parser.parse_args().run_dir)
