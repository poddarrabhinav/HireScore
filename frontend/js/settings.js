(function () {
  const SETTINGS_KEY = 'resume-scorer-ui-settings';
  const statusEl = document.getElementById('settings-status');
  const stage1El = document.getElementById('stage1-threshold');
  const stage2El = document.getElementById('stage2-threshold');
  const stage3El = document.getElementById('stage3-threshold');
  const embeddingEl = document.getElementById('embedding-profile');
  const llmEl = document.getElementById('use-llm');
  const saveBtn = document.getElementById('save-settings');
  const resetBtn = document.getElementById('reset-settings');
  const skillExpansionPathEl = document.getElementById('skill-expansion-path');
  const skillExpansionEditorEl = document.getElementById('skill-expansions-editor');
  const saveSkillExpansionsBtn = document.getElementById('save-skill-expansions');
  const reloadSkillExpansionsBtn = document.getElementById('reload-skill-expansions');

  let defaults = {
    stage1Threshold: 0.30,
    stage2Threshold: 0.30,
    stage3Threshold: 0.30,
    useLlm: true,
    embeddingProfile: 'small',
    embeddingOptions: [
      { value: 'small', label: 'Small' },
      { value: 'medium', label: 'Medium' },
    ],
  };

  function showStatus(message, tone) {
    statusEl.hidden = false;
    statusEl.textContent = message;
    statusEl.className = `settings-status ${tone || ''}`.trim();
  }

  function readStored() {
    try {
      const raw = window.localStorage.getItem(SETTINGS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) {
      return {};
    }
  }

  function writeStored(payload) {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
  }

  function clamp(value, min, max, fallback) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) return fallback;
    return Math.max(min, Math.min(max, numeric));
  }

  function populateOptions(options, selectedValue) {
    embeddingEl.innerHTML = '';
    options.forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.label;
      if (opt.value === selectedValue) option.selected = true;
      embeddingEl.appendChild(option);
    });
  }

  function applySettings(values) {
    stage1El.value = values.stage1Threshold.toFixed(2);
    stage2El.value = values.stage2Threshold.toFixed(2);
    stage3El.value = values.stage3Threshold.toFixed(2);
    llmEl.checked = !!values.useLlm;
    populateOptions(values.embeddingOptions || defaults.embeddingOptions, values.embeddingProfile);
  }

  async function loadSkillExpansions() {
    try {
      const res = await fetch('/api/skill-expansions');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      skillExpansionPathEl.textContent = `Editing ${data.path || 'skill_expansions.yaml'}`;
      skillExpansionEditorEl.value = data.content || '';
    } catch (err) {
      showStatus(`Could not load skill expansions: ${err.message}`, 'warning');
    }
  }

  async function loadDefaults() {
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      defaults = {
        stage1Threshold: Number(data.stage1_threshold ?? 0.30),
        stage2Threshold: Number(data.stage2_threshold ?? 0.30),
        stage3Threshold: Number(data.stage3_threshold ?? 0.30),
        useLlm: Boolean(data.use_llm ?? true),
        embeddingProfile: data.embedding_profile || 'small',
        embeddingOptions: Array.isArray(data.embedding_options) && data.embedding_options.length
          ? data.embedding_options
          : defaults.embeddingOptions,
      };
    } catch (_) {
      showStatus('Using built-in defaults because the settings API was unavailable.', 'warning');
    }

    const stored = readStored();
    applySettings({ ...defaults, ...stored, embeddingOptions: defaults.embeddingOptions });
  }

  saveBtn.addEventListener('click', function () {
    const payload = {
      stage1Threshold: clamp(stage1El.value, 0, 0.95, defaults.stage1Threshold),
      stage2Threshold: clamp(stage2El.value, 0, 0.95, defaults.stage2Threshold),
      stage3Threshold: clamp(stage3El.value, 0, 0.95, defaults.stage3Threshold),
      useLlm: !!llmEl.checked,
      embeddingProfile: embeddingEl.value || defaults.embeddingProfile,
    };
    writeStored(payload);
    stage1El.value = payload.stage1Threshold.toFixed(2);
    stage2El.value = payload.stage2Threshold.toFixed(2);
    stage3El.value = payload.stage3Threshold.toFixed(2);
    showStatus('Settings saved locally. The next scoring run will use these values.', 'success');
  });

  resetBtn.addEventListener('click', function () {
    writeStored({
      stage1Threshold: defaults.stage1Threshold,
      stage2Threshold: defaults.stage2Threshold,
      stage3Threshold: defaults.stage3Threshold,
      useLlm: defaults.useLlm,
      embeddingProfile: defaults.embeddingProfile,
    });
    applySettings(defaults);
    showStatus('Settings reset to backend defaults.', 'success');
  });

  saveSkillExpansionsBtn.addEventListener('click', async function () {
    try {
      const res = await fetch('/api/skill-expansions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: skillExpansionEditorEl.value }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      skillExpansionPathEl.textContent = `Editing ${data.path || 'skill_expansions.yaml'}`;
      showStatus('Skill expansions saved. Stage 1 will use the updated map on the next run.', 'success');
    } catch (err) {
      showStatus(`Could not save skill expansions: ${err.message}`, 'warning');
    }
  });

  reloadSkillExpansionsBtn.addEventListener('click', async function () {
    await loadSkillExpansions();
    showStatus('Reloaded skill expansions from disk.', 'success');
  });

  loadDefaults();
  loadSkillExpansions();
})();
