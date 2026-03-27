(function () {
  const statusEl = document.getElementById('eval-status');
  const jdTextEl = document.getElementById('eval-jd-text');
  const jdFileEl = document.getElementById('eval-jd-file');
  const valuesFileEl = document.getElementById('eval-values-file');
  const zipFileEl = document.getElementById('eval-zip-file');
  const labelsFileEl = document.getElementById('eval-labels-file');
  const runBtn = document.getElementById('run-eval');
  const resultsEl = document.getElementById('eval-results');
  const metricsEl = document.getElementById('eval-metrics');
  const comparisonsEl = document.getElementById('eval-comparisons');

  function showStatus(message, tone) {
    statusEl.hidden = false;
    statusEl.textContent = message;
    statusEl.className = `settings-status ${tone || ''}`.trim();
  }

  function fmtScore(value) {
    return typeof value === 'number' ? value.toFixed(2) : '—';
  }

  function renderMetrics(metrics, payload) {
    metricsEl.innerHTML = '';
    const cards = [
      ['Matched', payload.matched_count],
      ['Resumes', payload.resume_count],
      ['Targets', payload.target_count],
      ['MAE', fmtScore(metrics.mae)],
      ['RMSE', fmtScore(metrics.rmse)],
      ['<= 0.10', metrics.within_0_10],
      ['<= 0.20', metrics.within_0_20],
    ];
    cards.forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'stat-card';
      card.innerHTML = `<div class="stat-val">${value}</div><div class="stat-lbl">${label}</div>`;
      metricsEl.appendChild(card);
    });
  }

  function renderComparisons(items) {
    comparisonsEl.innerHTML = '';
    items.forEach((item) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${item.filename}</td>
        <td>${fmtScore(item.predicted_score)}</td>
        <td>${fmtScore(item.target_score)}</td>
        <td>${fmtScore(item.absolute_error)}</td>
        <td>${item.rank ?? '—'}</td>
        <td>${item.stage_eliminated ? `Stage ${item.stage_eliminated}` : 'Passed'}</td>
      `;
      comparisonsEl.appendChild(tr);
    });
  }

  runBtn.addEventListener('click', async function () {
    if (!labelsFileEl.files[0]) {
      showStatus('Upload a JSON file with target scores.', 'warning');
      return;
    }
    if (!zipFileEl.files[0]) {
      showStatus('Upload a ZIP of resumes.', 'warning');
      return;
    }
    if (jdTextEl.value.trim() && jdFileEl.files[0]) {
      showStatus('Use either JD text or a JD file for evaluation, not both.', 'warning');
      return;
    }
    if (!jdTextEl.value.trim() && !jdFileEl.files[0]) {
      showStatus('Provide either JD text or a JD file.', 'warning');
      return;
    }

    const form = new FormData();
    form.append('labels_file', labelsFileEl.files[0]);
    form.append('resumes_zip', zipFileEl.files[0]);
    form.append('job_description', jdTextEl.value);
    if (jdFileEl.files[0]) form.append('jd_file', jdFileEl.files[0]);
    if (valuesFileEl.files[0]) form.append('values_file', valuesFileEl.files[0]);
    showStatus('Running evaluation…', 'success');

    try {
      const res = await fetch('/api/eval/run', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      renderMetrics(payload.metrics, payload);
      renderComparisons(payload.comparisons || []);
      resultsEl.hidden = false;
      showStatus('Evaluation complete.', 'success');
    } catch (err) {
      showStatus(`Evaluation failed: ${err.message}`, 'warning');
    }
  });
})();
