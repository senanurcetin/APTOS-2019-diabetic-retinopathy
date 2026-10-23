'use strict';
const evidence = JSON.parse(document.getElementById('evidence').textContent);
const percent = value => (value * 100).toFixed(1) + '%';
const put = (id, html) => { document.getElementById(id).innerHTML = html; };
put('gap', evidence.external_sensitivity_gap_percentage_points.toFixed(2));
function draw() {
  const name = document.getElementById('cohort').value;
  const row = evidence.external.find(x => x.dataset === name);
  const grade = evidence.grade_reports.find(x => x.dataset === name);
  const operating = evidence.operating_points.find(x => x.cohort === name);
  put('metrics', [['External eyes', row.n], ['Referable eyes', row.referable], ['Missed at grade ≥ 2', row.missed_referable], ['Specificity at grade ≥ 2', percent(row.specificity)]].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join(''));
  put('bars', [[`Grade ≥ 2 · cut ${grade.thresholds[1].toFixed(3)}`, row.sensitivity, row.sensitivity_wilson95], [`APTOS operating point · cut ${evidence.operating_cut.toFixed(3)}`, operating.sensitivity, operating.sensitivity_wilson95]].map(([label, value, interval], i) => `<div class="bar-label"><span>${label}</span><b>${percent(value)}</b></div><div class="bar-track"><div class="bar ${i ? 'alt' : ''}" style="width:${value * 100}%"></div></div><div class="interval">95% interval ${percent(interval[0])}–${percent(interval[1])}</div>`).join(''));
  put('grades', ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'].map((label, i) => `<tr><td>${i} / ${label}</td><td>${grade.true_distribution[i]}</td><td>${percent(grade.per_class_recall[i])}</td></tr>`).join(''));
}
put('cvrows', evidence.cross_validation.map(row => `<tr><td>${row.variant}</td><td><code>${row.sweep}</code></td><td>${row.folds}</td><td>${row.fold_test_qwk_mean.toFixed(4)} ± ${row.fold_test_qwk_sd.toFixed(4)}</td><td>${row.ensemble_test_qwk.toFixed(4)}</td></tr>`).join(''));
put('hashes', Object.entries(evidence.inputs_sha256).map(([name, hash]) => `<p>${name}<code>${hash}</code></p>`).join(''));
document.getElementById('cohort').addEventListener('change', draw);
document.getElementById('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(evidence, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'aptos-research-evidence.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
draw();
