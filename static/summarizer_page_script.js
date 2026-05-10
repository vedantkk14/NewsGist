// Theme Management System - Default to light mode
class ThemeManager {
  constructor() {
    this.themeToggle = document.getElementById("themeToggle");
    this.init();
  }

  init() {
    const defaultTheme = "light";
    this.setTheme(defaultTheme);
    this.themeToggle.checked = false;

    this.themeToggle.addEventListener("change", () => {
      const newTheme = this.themeToggle.checked ? "dark" : "light";
      this.setTheme(newTheme);
    });
  }

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("news-analyzer-theme", theme);
    this.updateInternalLinks(theme);
  }

  updateInternalLinks(theme) {
    const internalLinks = document.querySelectorAll('a[href^="/"], a[href^="./"], a[href^="../"]');
    internalLinks.forEach((link) => {
      const url = new URL(link.href, window.location.origin);
      url.searchParams.set("theme", theme);
      link.href = url.toString();
    });
  }
}

// Initialize theme management
const themeManager = new ThemeManager();

// Get DOM elements
const inputText = document.getElementById("inputText");
const charCount = document.getElementById("charCount");
const resetBtn = document.getElementById("resetBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const outputArea = document.getElementById("outputArea");
const modelOptions = document.querySelectorAll(".model-option");

let currentSummary = "";
let currentSpeech = null;
let selectedModel = "default";
let originalTextLength = 0;

// Check for TTS support
const ttsSupported = "speechSynthesis" in window;

// Update character count and button states
function updateUI() {
  const textLength = inputText.value.length;
  originalTextLength = textLength;
  charCount.textContent = textLength.toLocaleString() + " characters";

  if (textLength > 0) {
    charCount.classList.add("active");
  } else {
    charCount.classList.remove("active");
  }

  const hasText = inputText.value.trim().length > 0;
  resetBtn.disabled = !hasText;
  summarizeBtn.disabled = !hasText;
}

// Model selection handler
modelOptions.forEach((option) => {
  option.addEventListener("click", () => {
    modelOptions.forEach((opt) => opt.classList.remove("active"));
    option.classList.add("active");
    selectedModel = option.dataset.model;
    if (inputText.value.trim()) {
      summarizeBtn.click();
    }
  });
});

// Text summarization API call
async function summarizeText(text, model = "default") {
  try {
    const endpoint = model === "pegasus" ? "/summarize_pegasus" : "/summarize";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    const data = await response.json();
    return data.summary || data.error || "Unable to generate summary";
  } catch (error) {
    return "Error: Unable to connect to summarization service.";
  }
}

// Show loading state
function showLoading() {
  outputArea.innerHTML = `
    <div class="loading-container">
      <div class="loading-status">
        <div class="loading-pulse"></div>
        <span style="font-size: 0.9rem; font-weight: 500; color: var(--text-secondary);">Analyzing article...</span>
      </div>
      <div class="skeleton-text long"></div>
      <div class="skeleton-text medium"></div>
      <div class="skeleton-text short"></div>
    </div>
  `;
}

// Show summary result
function showSummary(summary) {
  currentSummary = summary;
  const summaryLength = summary.length;
  const reduction = originalTextLength - summaryLength;
  const reductionPercentage = originalTextLength > 0 ? Math.round((reduction / originalTextLength) * 100) : 0;
  const escapedSummary = summary.replace(/'/g, "\\'").replace(/"/g, "&quot;");

  outputArea.classList.add("has-content");
  outputArea.innerHTML = `
    <div style="width: 100%;">
      <div class="summary-box">
        <div class="summary-content">
          <p class="summary-text">${summary}</p>
        </div>
        <div class="summary-stats">
          <div class="stat-item">
            <span class="stat-value">${summaryLength.toLocaleString()}</span>
            <div class="stat-label">Characters</div>
          </div>
          <div class="stat-item">
            <span class="stat-value">-${reduction.toLocaleString()}</span>
            <div class="stat-label">Reduced</div>
          </div>
          <div class="stat-item">
            <span class="stat-value">${reductionPercentage}%</span>
            <div class="stat-label">Compression</div>
          </div>
        </div>
        <div class="action-buttons">
          <button onclick="copySummary('${escapedSummary}')" class="btn btn-copy">Copy Text</button>
          <button onclick="analyzeSentiment('${escapedSummary}')" class="btn btn-sentiment">Sentiment Analysis</button>
          <button onclick="listenToSummary('${escapedSummary}')" class="btn btn-listen" ${!ttsSupported ? "disabled" : ""}>Listen</button>
        </div>
      </div>
    </div>
  `;
}

// Show error state
function showError(error) {
  outputArea.innerHTML = `
    <div style="width: 100%; text-align: center;">
      <div style="color: #e53e3e; margin-bottom: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem; animation: shake 0.5s ease-in-out;">❌</div>
        <p>Error: ${error}</p>
      </div>
      <button onclick="showEmptyState()" class="btn btn-secondary">🔄 Try Again</button>
    </div>
  `;
}

// Show empty state
function showEmptyState() {
  outputArea.classList.remove("has-content");
  currentSummary = "";
  originalTextLength = 0;
  if (currentSpeech) {
    speechSynthesis.cancel();
    currentSpeech = null;
  }
  outputArea.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">📄</div>
      <p>Your summary will appear here</p>
    </div>
  `;
}

// Copy summary to clipboard
function copySummary(text) {
  const btn = event.currentTarget;
  navigator.clipboard.writeText(text).then(() => {
    const originalText = btn.innerHTML;
    btn.innerHTML = "✅ Copied!";
    setTimeout(() => { btn.innerHTML = originalText; }, 2000);
  });
}

// Analyze Sentiment (Redirect version)
function analyzeSentiment(text) {
    // Get current theme to pass it along
    const theme = document.documentElement.getAttribute('data-theme') || 'light';
    
    // Create a dynamic form to submit to the sentiment analysis route
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = `/sentiment_analysis?theme=${theme}`;
    form.style.display = 'none';

    const textInput = document.createElement('input');
    textInput.type = 'hidden';
    textInput.name = 'summary_text';
    textInput.value = text;

    form.appendChild(textInput);
    document.body.appendChild(form);
    form.submit();
}

// Listen to Summary (TTS)
function listenToSummary(text) {
    const btn = event.currentTarget;
    if (currentSpeech && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        currentSpeech = null;
        btn.innerHTML = "Listen";
        return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => { btn.innerHTML = "Listen"; currentSpeech = null; };
    btn.innerHTML = "Stop Listening";
    currentSpeech = utterance;
    window.speechSynthesis.speak(utterance);
}

// Event listeners
inputText.addEventListener("input", updateUI);

resetBtn.addEventListener("click", () => {
  inputText.value = "";
  updateUI();
  showEmptyState();
});

summarizeBtn.addEventListener("click", async () => {
  if (!inputText.value.trim()) return;
  showLoading();
  try {
    const summary = await summarizeText(inputText.value, selectedModel);
    if (summary && summary.trim()) {
      showSummary(summary);
    } else {
      showError("Unable to generate summary");
    }
  } catch (error) {
    showError("Something went wrong");
  }
});

// Initialize UI
updateUI();
showEmptyState();
