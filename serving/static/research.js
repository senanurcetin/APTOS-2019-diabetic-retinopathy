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

const followup = evidence.followup;
const contrast = followup.paired_auc.contrasts.adjudicated_minus_control;
const signed = value => `${value >= 0 ? '+' : ''}${value.toFixed(4)}`;
put('followup-metrics', [['Test eyes', followup.n_test_eyes], ['Examinations', followup.n_test_examinations], ['AUC difference: labels − control', signed(contrast.difference)], ['Missed referrals: labels', followup.fixed_original_cuts.adjudicated.referable.fn]].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join(''));
put('followup-interval', `Primary AUC difference ${signed(contrast.difference)} · 95% paired examination interval [${signed(contrast.ci_lower)}, ${signed(contrast.ci_upper)}]. 2,000 draws, seed 42.`);
put('followup-rows', ['baseline', 'adjudicated', 'control'].map(arm => {
  const row = followup.fixed_original_cuts[arm];
  return `<tr><td>${arm}</td><td>${row.auc.toFixed(4)}</td><td>${row.qwk.toFixed(4)}</td><td>${percent(row.referable.sensitivity)}</td><td>${percent(row.referable.specificity)}</td><td>${row.referable.fn}</td></tr>`;
}).join(''));
const retention = followup.aptos_retention.fixed_original_cuts;
put('followup-retention', `The gain has a transfer cost. On the reused 366-eye APTOS holdout, accuracy changed from ${percent(retention.baseline.accuracy)} to ${percent(retention.adjudicated.accuracy)}; QWK changed from ${retention.baseline.qwk.toFixed(4)} to ${retention.adjudicated.qwk.toFixed(4)}. No model was promoted.`);
document.getElementById('cohort').addEventListener('change', draw);
document.getElementById('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(evidence, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'aptos-research-evidence.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
draw();
