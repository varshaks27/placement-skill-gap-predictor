/**
 * main.js — SkillGap AI
 * Handles: AJAX role loading, study-hours slider, skill tag preview, form UX
 */

document.addEventListener("DOMContentLoaded", () => {

  /* ── 1. AJAX Role Loader ───────────────────────────────────────────────── */
  const companySelect = document.getElementById("company");
  const roleSelect    = document.getElementById("role");
  const roleLoader    = document.getElementById("roleLoader");

  if (companySelect && roleSelect) {
    companySelect.addEventListener("change", async () => {
      const company = companySelect.value;
      if (!company) return;

      // Show loader, disable role select
      roleSelect.disabled = true;
      roleSelect.innerHTML = '<option value="" disabled selected>Loading roles…</option>';
      if (roleLoader) roleLoader.classList.remove("d-none");

      try {
        const resp  = await fetch(`/get_roles?company=${encodeURIComponent(company)}`);
        const roles = await resp.json();

        roleSelect.innerHTML = '<option value="" disabled selected>— Select a role —</option>';
        roles.forEach(r => {
          const opt = document.createElement("option");
          opt.value       = r;
          opt.textContent = r;
          roleSelect.appendChild(opt);
        });

        roleSelect.disabled = roles.length === 0;
        if (roles.length === 0) {
          roleSelect.innerHTML = '<option value="" disabled selected>No roles found</option>';
        }
      } catch (err) {
        console.error("Role load error:", err);
        roleSelect.innerHTML = '<option value="" disabled selected>Error loading roles</option>';
      } finally {
        if (roleLoader) roleLoader.classList.add("d-none");
      }
    });
  }

  /* ── 2. Study Hours Slider ────────────────────────────────────────────── */
  const slider      = document.getElementById("study_hours");
  const hoursDisplay = document.getElementById("hoursDisplay");

  if (slider && hoursDisplay) {
    const updateSlider = () => {
      const val = parseInt(slider.value, 10);
      hoursDisplay.textContent = `${val} hr${val !== 1 ? "s" : ""}/week`;

      // Dynamic gradient fill on track
      const pct = ((val - slider.min) / (slider.max - slider.min)) * 100;
      slider.style.background =
        `linear-gradient(90deg, #6366f1 ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
    };
    slider.addEventListener("input", updateSlider);
    updateSlider(); // init
  }

  /* ── 3. Skill Tag Preview ─────────────────────────────────────────────── */
  const skillsTextarea  = document.getElementById("student_skills");
  const skillTagsContainer = document.getElementById("skillTags");

  if (skillsTextarea && skillTagsContainer) {
    const renderTags = () => {
      const raw   = skillsTextarea.value;
      const skills = raw.split(",").map(s => s.trim()).filter(s => s.length > 0);
      skillTagsContainer.innerHTML = "";
      skills.forEach(skill => {
        const span = document.createElement("span");
        span.className   = "skill-tag";
        span.textContent = skill.charAt(0).toUpperCase() + skill.slice(1);
        skillTagsContainer.appendChild(span);
      });
    };
    skillsTextarea.addEventListener("input", renderTags);
    renderTags(); // init in case of browser back-fill
  }

  /* ── 4. Submit Button Spinner ─────────────────────────────────────────── */
  const predictForm = document.getElementById("predictForm");
  const submitBtn   = document.getElementById("submitBtn");
  const btnText     = document.getElementById("btnText");
  const btnSpinner  = document.getElementById("btnSpinner");

  if (predictForm && submitBtn) {
    predictForm.addEventListener("submit", (e) => {
      // Basic client-side validation
      const company = document.getElementById("company")?.value;
      const role    = document.getElementById("role")?.value;
      const skills  = document.getElementById("student_skills")?.value.trim();

      if (!company || !role || !skills) return; // let browser handle required fields

      // Show loading state
      if (btnText)    btnText.textContent = "Analysing…";
      if (btnSpinner) btnSpinner.classList.remove("d-none");
      submitBtn.disabled = true;
    });
  }

  /* ── 5. Auto-dismiss alerts (if any) ─────────────────────────────────── */
  document.querySelectorAll(".alert-dismissible").forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      bsAlert?.close();
    }, 4000);
  });

  /* ── 6. Table sort highlight (history page) ───────────────────────────── */
  // Adds a subtle hover effect to the history table rows via CSS (no JS needed)
  // JS hook here for future extensibility

});
